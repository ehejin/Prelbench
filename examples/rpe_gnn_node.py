import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from examples.model import Model_PEARL
from examples.text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from examples.config import merge_config
import torch_geometric.transforms as T
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import HeteroData, Data


class transform_LAP():
    def __init__(self, instance=None):
        self.instance = None
    def __call__(self, hetero_data):
        node_mapping = {}  # Keep track of node indices from each type
        start_idx = 0
        total_num_nodes = 0
        # Step 1: Map node types to homogeneous indices
        for node_type in hetero_data.node_types:
            node_data = hetero_data[node_type]
            num_nodes = node_data['n_id'].size(0)
            node_mapping[node_type] = torch.arange(start_idx, start_idx + num_nodes)
            start_idx += num_nodes
            total_num_nodes += num_nodes

        # Step 2: Combine edge indices from all edge types
        all_edges = []

        for edge_type in hetero_data.edge_types:
            src_type, relation_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index

            # Map source and destination node indices to homogeneous space
            src_nodes = node_mapping[src_type][edge_index[0]]
            dst_nodes = node_mapping[dst_type][edge_index[1]]

            # Append to combined edge list
            all_edges.append(torch.stack([src_nodes, dst_nodes], dim=0))

        # Concatenate all edge indices into one edge index tensor
        if len(all_edges) > 0:
            all_edges = torch.cat(all_edges, dim=1)

        # Step 3: Create homogeneous Data object
        homogeneous_data = Data(num_nodes=total_num_nodes, edge_index=all_edges)

        # Step 4: Compute the Laplacian of the homogeneous graph
        edge_index, edge_weight = get_laplacian(homogeneous_data.edge_index, normalization='sym')
        laplacian = to_dense_adj(edge_index, edge_attr=edge_weight)

        hetero_data.edge_index = edge_index
        hetero_data.Lap = laplacian

        return hetero_data

'''class transform_LAP:
    def __init__(self, instance=None):
        self.instance = instance
    
    def __call__(self, instance):
        L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
        L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
        instance.Lap = L
        return data'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu_id", type=int, default=2)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--cfg", type=str, default=None, help="Path to PEARL cfg file")
args = parser.parse_args()
cfg = merge_config(args.cfg)

device = torch.device(f'cuda:{args.gpu_id}')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)


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

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    transform2 = T.Compose([table_input.transform, transform_LAP()])
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=transform2,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0
    )


import networkx as nx
import matplotlib.pyplot as plt

def visualize_hetero_graph(data):
    G = nx.Graph()  # Create an undirected graph (or use nx.DiGraph() for directed)
    print('doing node types')
    # Add nodes for each node type in HeteroData (like qualifying, drivers, results, etc.)
    for node_type in data.node_types:
        n_ids = data[node_type].n_id  # Get node IDs
        G.add_nodes_from([f"{node_type}_{i}" for i in n_ids], node_type=node_type)

    # Visualize edges by iterating over edge types
    print('doing edge tupes')
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src_type, relation, tgt_type = edge_type

        for i in range(edge_index.shape[1]):  # Loop over each edge
            src = edge_index[0, i]  # Source node
            tgt = edge_index[1, i]  # Target node
            G.add_edge(f"{src_type}_{src}", f"{tgt_type}_{tgt}", relation=relation)

    # Draw the graph using networkx and matplotlib
    print('drawing and saving fig')
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # You can use different layouts like spring_layout, circular_layout, etc.
    nx.draw(G, pos, with_labels=False, node_size=20, font_size=8, edge_color="gray", width=0.5)
    plt.savefig("./graph_visualization2.png")

def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        W_list = []
        for i in range(len(batch.Lap)):
            if cfg.BASIS:
                W = torch.eye(batch.Lap[i].shape[0]).to(device)
            else:
                W = torch.randn(batch.Lap[i].shape[0],cfg.num_samples).to(device) #BxNxM
            if len(W.shape) < 2:
                print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)
        #x_R = torch.randn((batch_size, 80, 1)).to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
            W_list
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        W_list = []
        for i in range(len(batch.Lap)):
            if cfg.BASIS:
                W = torch.eye(batch.Lap[i].shape[0]).to(device)
            else:
                W = torch.randn(batch.Lap[i].shape[0],cfg.num_samples).to(device) #BxNxM
            if len(W.shape) < 2:
                print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)
        pred = model(
            batch,
            task.entity_table,
            W
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model_PEARL(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
    cfg=cfg
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")