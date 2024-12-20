import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from examples.model import MODEL_LINK, Model_PEARL, Model_SIGNNET
from examples.text_embedder import GloveTextEmbedding
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import HeteroData, Data

from examples.config import merge_config

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
from torch_geometric.utils import to_scipy_sparse_matrix
import wandb


class transform_LAP():
    '''
    This class transforms 
    '''
    def __init__(self, instance=None, PE1=True, device=None):
        self.instance = None
        self.PE1 = PE1
    def __call__(self, hetero_data):
        node_mapping = {}  # Keep track of node indices from each type
        start_idx = 0
        total_num_nodes = 0
        reverse_node_mapping = {} 
        for node_type in hetero_data.node_types:
            node_data = hetero_data[node_type]
            num_nodes = node_data['n_id'].size(0)
            node_mapping[node_type] = torch.arange(start_idx, start_idx + num_nodes)
            for i in range(num_nodes):
                reverse_node_mapping[start_idx + i] = (node_type, i)
            start_idx += num_nodes
            total_num_nodes += num_nodes
        all_edges = []

        for edge_type in hetero_data.edge_types:
            src_type, relation_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index
            src_nodes = node_mapping[src_type][edge_index[0]]
            dst_nodes = node_mapping[dst_type][edge_index[1]]
            all_edges.append(torch.stack([src_nodes, dst_nodes], dim=0))
        if len(all_edges) > 0:
            all_edges = torch.cat(all_edges, dim=1)
        homogeneous_data = Data(num_nodes=total_num_nodes, edge_index=all_edges)
        edge_index = homogeneous_data.edge_index

        if not self.PE1:
            edge_index, edge_weight = get_laplacian(homogeneous_data.edge_index, normalization='sym')
            laplacian = to_dense_adj(edge_index, edge_attr=edge_weight)
            hetero_data.Lap = laplacian

        hetero_data.edge_index = edge_index
        hetero_data.num_nodes = total_num_nodes
        hetero_data.reverse_node_mapping = reverse_node_mapping

        return hetero_data


def sparse_evd(laplacian, k, device, smallest):
    try: 
        sparse_laplacian = laplacian  # Create a sparse CSR matrix

        # Compute the largest k eigenvalues and eigenvectors
        eigvals, eigvecs = eigsh(sparse_laplacian, k=k, which='SM', maxiter=5000000)  # 'LM' = Largest Magnitude

        # Convert the results back to PyTorch tensors
        eigvals = torch.tensor(eigvals, device=device, dtype=torch.float32)
        eigvecs = torch.tensor(eigvecs, device=device, dtype=torch.float32)
    except:
        print("Eigsh failed!!")
        dense_laplacian = torch.tensor(laplacian.toarray(), device=device, dtype=torch.float32)
        eigvals, eigvecs = torch.linalg.eigh(dense_laplacian)
        if not smallest:
            eigvals = eigvals[-k:]  # Largest eigenvalues
            eigvecs = eigvecs[:, -k:]
        else:
            eigvals = eigvals[:k] # SMALLEST
            eigvecs = eigvecs[:, :k]
    
    return eigvals, eigvecs

class transform_LAP_OG():
    def __init__(self, instance=None, PE1=True, pe_dims=8, device=None, smallest=True):
        self.instance = instance
        self.PE1 = PE1
        self.pe_dims = pe_dims  # Number of smallest eigenvectors to extract
        self.device = device
        if smallest:
            print("PRINT SMALLEST EIGS")
        else:
            print("PRINT LARGEST EIGS")
        self.smallest = smallest

    def __call__(self, hetero_data):
        node_mapping = {}  # Keep track of node indices from each type
        reverse_node_mapping = {} 
        start_idx = 0
        total_num_nodes = 0
        
        # Map heterogeneous nodes to a homogeneous index space
        for node_type in hetero_data.node_types:
            node_data = hetero_data[node_type]
            num_nodes = node_data['n_id'].size(0)
            node_mapping[node_type] = torch.arange(start_idx, start_idx + num_nodes)
            for i in range(num_nodes):
                reverse_node_mapping[start_idx + i] = (node_type, i)
            start_idx += num_nodes
            total_num_nodes += num_nodes

        # Collect all edges into a homogeneous graph
        all_edges = []
        for edge_type in hetero_data.edge_types:
            src_type, relation_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index
            src_nodes = node_mapping[src_type][edge_index[0]]
            dst_nodes = node_mapping[dst_type][edge_index[1]]
            all_edges.append(torch.stack([src_nodes, dst_nodes], dim=0))

        # Combine all edge indices
        if len(all_edges) > 0:
            all_edges = torch.cat(all_edges, dim=1)
        homogeneous_data = Data(num_nodes=total_num_nodes, edge_index=all_edges)

        edge_index, edge_weight = get_laplacian(homogeneous_data.edge_index, normalization='sym')
        laplacian_sparse = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=total_num_nodes)
        #laplacian = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=total_num_nodes).squeeze(0)
        
        eigenvalues, eigenvectors = sparse_evd(laplacian_sparse, k=self.pe_dims, device=self.device, smallest=self.smallest) 
        
        #d = min(self.pe_dims, total_num_nodes)
        smallest_eigenvalues = eigenvalues #eigenvalues[:d]
        smallest_eigenvectors = eigenvectors #eigenvectors[:, :d]
        
        # Store the Laplacian, eigenvalues, and eigenvectors in hetero_data
        #hetero_data.Lap = laplacian
        hetero_data.Lambda = smallest_eigenvalues
        hetero_data.V = smallest_eigenvectors

        # Update hetero_data with homogeneous graph information
        hetero_data.edge_index = homogeneous_data.edge_index
        hetero_data.num_nodes = total_num_nodes
        hetero_data.reverse_node_mapping = reverse_node_mapping

        return hetero_data


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument('--name', type=str, default=None)
parser.add_argument("--cfg", type=str, default=None, help="Path to PEARL cfg file")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()



device = torch.device(f'cuda:{args.gpu_id}') #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

cfg = merge_config(args.cfg)

if args.wandb:
    run = wandb.init(config=cfg, project='Relbench-LINK', name=args.name)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    transform = transform_LAP(PE1=cfg.PE1, device=device, smallest=cfg.smallest)
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        transform=transform
    )

model = Model_LINK(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
    cfg=cfg,
    device=device,
    PE1=False,
    REL=False
).to(device)

print("TOTAL NUM PARAMS: ", sum(p.numel() for p in model.parameters()))

'''model = MODEL_LINK(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
    cfg=cfg,
    device=device
).to(device)'''

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    i = 0
    mult_subgraphs = []
    mult_laps = []
    for batch in tqdm(loader_dict["train"], total=total_steps):
        del batch.Lap
        mult_subgraphs.append(batch)
        mult_laps.append(batch.Lap)
        if i == 5:
            batch = batch.to(device)
            out = model.forward_dst_readout(
                mult_subgraphs, task.src_entity_table, task.dst_entity_table, mult_laps
            ).flatten()

            batch_size = batch[task.src_entity_table].batch_size

            # Get ground-truth
            input_id = batch[task.src_entity_table].input_id
            src_batch, dst_index = train_sparse_tensor[input_id]

            # Get target label
            target = torch.isin(
                batch[task.dst_entity_table].batch
                + batch_size * batch[task.dst_entity_table].n_id,
                src_batch + batch_size * dst_index,
            ).float()

            # Optimization
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss.backward()

            optimizer.step()

            loss_accum += float(loss) * out.numel()
            count_accum += out.numel()

            steps += 1
            if steps > args.max_steps_per_epoch:
                break

            mult_subgraphs = []
        i += 1

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list: list[Tensor] = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        test_pred = test(loader_dict["test"])
        test_metrics = task.evaluate(test_pred)
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )
        val_metrics['Train loss'] = train_loss
        val_metrics['test_MAP'] = test_metrics['link_prediction_map']
        wandb.log(val_metrics)

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")