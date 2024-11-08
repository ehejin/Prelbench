import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='6'
import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from examples.model import Model, Model_PEARL
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
import wandb

WANDB=False
CROSSVAL = False

class transform_LAP():
    def __init__(self, instance=None, PE1=True):
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


class transform_LAP_OG():
    def __init__(self, instance=None, PE1=True):
        self.instance = instance
        self.PE1 = PE1
        self.relation_laplacians = {}  
        self.relation_mappings = {}  

    def __call__(self, hetero_data):
        print(hetero_data)
        node_mapping = {}
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

        hetero_data.num_nodes = total_num_nodes
        hetero_data.reverse_node_mapping = reverse_node_mapping

        for edge_type in hetero_data.edge_types:
            src_type, relation_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index

            src_nodes = node_mapping[src_type][edge_index[0]]
            dst_nodes = node_mapping[dst_type][edge_index[1]]
            edges = torch.stack([src_nodes, dst_nodes], dim=0)

            relation_data = Data(num_nodes=total_num_nodes, edge_index=edges)

            if not self.PE1:
                edge_index, edge_weight = get_laplacian(relation_data.edge_index, normalization='sym')
                laplacian = to_dense_adj(edge_index, edge_attr=edge_weight)
                self.relation_laplacians[relation_type] = laplacian

            self.relation_mappings[relation_type] = {
                "edge_index": edges,
                "laplacian": self.relation_laplacians.get(relation_type, None),
                "num_nodes": total_num_nodes,
                "reverse_node_mapping": reverse_node_mapping,
            }
        hetero_data.relation_mappings = self.relation_mappings

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
parser.add_argument("--name", type=str, default="RPE")
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--cfg", type=str, default=None, help="Path to PEARL cfg file")
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--n_phi_layers", type=int, default=6)
parser.add_argument("--hidden_phi_layers", type=int, default=40)
parser.add_argument("--pe_dims", type=int, default=37)

args = parser.parse_args()
cfg = merge_config(args.cfg)

#cfg.RAND_k = args.k
#cfg.n_phi_layers = args.n_phi_layers
#cfg.hidden_phi_layers = args.hidden_phi_layers
#cfg.pe_dims = args.pe_dims

#new_name = 'k=' + str(args.k) + '_' + str(args.n_phi_layers) + ':' + str(args.hidden_phi_layers) + ',' + str(args.pe_dims)
if WANDB:
    run = wandb.init(config=cfg, project='Relbench', name=args.name)

device = torch.device(f'cuda:{args.gpu_id}')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

db = dataset.get_db()
print(db.min_timestamp, db.max_timestamp)

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
import pdb; pdb.set_trace()

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


if CROSSVAL:
    loader_dict=None
else:
    loader_dict: Dict[str, NeighborLoader] = {}
    for split in ["train", 'val', "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        if split != 'test':
            transform2 = T.Compose([table_input.transform, transform_LAP(PE1=cfg.PE1)])
        else:
            transform2 = transform_LAP(PE1=cfg.PE1)
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
        import pdb; pdb.set_trace()
        last_target_nodes = table_input.nodes[-2:]


for batch in loader_dict['test']:
    if batch.tf_dict['drivers'].num_rows != 100:
        print()