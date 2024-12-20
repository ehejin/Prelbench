from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet 
from relbench.modeling.resnet import ResNet as ResNet2
from torch_geometric.nn import LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType 
from relbench.modeling.heteroConv import HeteroConv as HeteroConv2
from torch_geometric.nn.conv import HeteroConv
from relbench.modeling.pe import GINPhi
from relbench.modeling.mlp import MLP as MLP2
from torch_geometric.nn import GINEConv

class HeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroEncoder_PEARL(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet2,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        PE_dict
    ) -> Dict[NodeType, Tensor]:
        x_dict = {}
        for node_type, tf in tf_dict.items():
            if node_type not in PE_dict:
                PE_dict[node_type] = None
            x_dict[node_type] = self.encoders[node_type](tf, PE_dict[node_type])
        return x_dict
    
    '''def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        PE_dict_per_relation
    ):
    
        x_dict = {}

        for relation_type, PE_dict in PE_dict_per_relation.items():
            
            for node_type, tf in tf_dict.items():
                if node_type not in PE_dict:
                    PE_dict[node_type] = None
                
                if node_type not in x_dict:
                    x_dict[node_type] = 0
                x_dict[node_type] += self.encoders[node_type](tf, PE_dict[node_type])
            
            #x_dict_per_relation[relation_type] = x_dict

        return x_dict[node_type] #x_dict_per_relation'''


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )

    def reset_parameters(self):
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphSAGE_PEARL(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        pe_emb=37,
        create_mlp=None
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
                create_mlp=create_mlp,
                pe_emb=pe_emb
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        PE,
        reverse_node_mapping,
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(PE, reverse_node_mapping, x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            new_x_dict = {key: x.detach().relu() for key, x in x_dict.items()}
            x_dict = new_x_dict#{key: x.relu() for key, x in x_dict.items()}

        return x_dict


def build_edge_attr_dict(PE, reverse_node_mapping, hetero_data):
    """
    Build the edge_attr_dict using the learned positional encodings.

    Args:
        PE (torch.Tensor): Learned positional encodings of shape (N, N, K).
        reverse_node_mapping (dict): Mapping from homogeneous index to (node_type, node_idx).
        hetero_data (HeteroData): The heterogeneous graph data object.

    Returns:
        edge_attr_dict (dict): Dictionary of edge attributes for each edge type.
    """
    edge_attr_dict = {}

    # Iterate over each edge type in the heterogeneous graph
    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        src_list, dst_list = edge_index
        edge_attr_list = []

        # Iterate over each edge in the edge index list
        for src_hom, dst_hom in zip(src_list, dst_list):
            # Map homogeneous indices to heterogeneous indices
            src_node_type, src_node_idx = reverse_node_mapping[src_hom.item()]
            dst_node_type, dst_node_idx = reverse_node_mapping[dst_hom.item()]

            # Check if source and destination nodes match the current edge type
            if src_node_type == edge_type[0] and dst_node_type == edge_type[2]:
                # Extract the positional encoding for the current edge
                edge_attr = PE[src_hom, dst_hom, :]  # Shape: (K,)
                edge_attr_list.append(edge_attr)
        
        # Convert the list of edge attributes to a tensor and store in the dictionary
        if edge_attr_list:
            edge_attr_dict[edge_type] = torch.stack(edge_attr_list)
        else:
            edge_attr_dict[edge_type] = None

    return edge_attr_dict



class HeteroGraphSAGE_LINK(HeteroGraphSAGE):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        cfg=None
    ):
        super().__init__(
            node_types=node_types,
            edge_types=edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers
        )
        self.cfg = cfg

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            convs = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum"
            )
            self.convs.append(convs)
        
        self.phi = torch.nn.ModuleList()
        self.pe_embedding = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            new_phi = GINPhi(1, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn, RAND_LAP=False, pooling=True)
            new_emb = torch.nn.Linear(self.cfg.pe_dims+128, 128) #self.create_mlp(self.cfg.pe_dims+128, 128)
            self.phi.append(new_phi)
            self.pe_embedding.append(new_emb)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        self.MP.reset_parameters()

    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None):
        return MLP2(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        PE,
        reverse_mapping,
        edge_index,
        hetero_data
    ) -> Dict[NodeType, Tensor]:
        reverse_node_mapping = reverse_mapping
        new_list = [PE]

        for _, (conv, norm_dict, phi, pe_embedding) in enumerate(zip(self.convs, self.norms, self.phi, self.pe_embedding)):
            #x_dict = conv(x_dict, edge_index_dict, PE=PE, reverse_node_mapping=reverse_mapping, edge_index=edge_index)
            PE = phi(new_list, edge_index, self.cfg.BASIS, running_sum=False, final=False) 
            #PE = pe_embedding(PE)

            for homogeneous_idx, pos_encoding in enumerate(PE):
                node_type, node_idx = reverse_node_mapping[homogeneous_idx]
                x_dict[node_type][node_idx] = pe_embedding(torch.cat((x_dict[node_type][node_idx], pos_encoding), dim=-1))

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.detach().relu() for key, x in x_dict.items()}

        return x_dict
