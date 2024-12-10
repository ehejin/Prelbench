from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroGraphSAGE_LINK, HeteroEncoder_PEARL, HeteroGraphSAGE_PEARL, HeteroTemporalEncoder
#from relbench.modeling.pe import GINPhi
from relbench.modeling.mlp import MLP as MLP2
from relbench.modeling.pe import K_PEARL_PE, GINPhi, MaskedSignInvPe, SignInvPe
import numpy as np
from relbench.examples.model import Model


class GIN(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False, laplacian=None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            new_mlp = create_mlp(in_dims, hidden_dims)
            layer = GINLayer(new_mlp)
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
        new_m = create_mlp(hidden_dims, out_dims)
        layer = GINLayer(new_m)
        self.layers.append(layer)
        self.laplacian=laplacian
        print("GINPHI LAP is: ", laplacian)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index, laplacian=self.laplacian, mask=mask)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            if mask is not None:
                X[~mask] = 0
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if mask is None:
                    if X.ndim == 3:
                        X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                    else:
                        X = self.batch_norms[i](X)
                else:
                    X[mask] = self.batch_norms[i](X[mask])
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True) #torch.empty(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, laplacian=False, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """

        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z, mask)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims


class MODEL_LINK(Model):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        cfg=None,
        device=None
    ):
        super().__init__(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            out_channels=out_channels,
            aggr=aggr,
            norm=norm,
            shallow_list=shallow_list,
            id_awareness=id_awareness,
        )

        self.gnn = HeteroGraphSAGE_LINK(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
            cfg=cfg
        )

        self.device = device

        self.cfg = cfg

        Phi = GINPhi(self.cfg.n_phi_layers, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn, RAND_LAP=False)
                    
        self.positional_encoding = K_PEARL_PE(Phi, cfg.BASIS, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, 
                            mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)

        self.num_samples = cfg.num_samples

        #self.pe_embedding = torch.nn.Linear(self.positional_encoding.out_dims, self.cfg.node_emb_dims)
    
    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None) -> MLP:
        print(in_dims)
        return MLP2(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        for k in range(5):
            W_list = []
            for i in range(len(batch.Lap)):
                if self.cfg.BASIS:
                    W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                else:
                    W = torch.randn(batch.Lap[i].shape[0],self.num_samples//5).to(self.device) #BxNxM
                W_list.append(W)
            if k < 4:
                with torch.no_grad():
                    self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
            else:
                PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
            del W_list, W
            torch.cuda.empty_cache()
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict, 
            batch.edge_index_dict,
            PE, 
            batch.reverse_node_mapping,
            batch.edge_index,
            batch
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])


    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time

        for k in range(5):
            W_list = []
            for i in range(len(batch.Lap)):
                if self.cfg.BASIS:
                    W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                else:
                    W = torch.randn(batch.Lap[i].shape[0],self.num_samples//5).to(self.device) #BxNxM
                W_list.append(W)
            if k < 4:
                with torch.no_grad():
                    self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
            else:
                PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
            del W_list, W
            torch.cuda.empty_cache()

        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            PE, 
            batch.reverse_node_mapping,
            batch.edge_index,
            batch
        )

        return self.head(x_dict[dst_table])





class Model_SIGNNET(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        cfg=None,
        PE1=True,
        device=None,
        REL=False
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = HeteroEncoder_PEARL(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers
            #create_mlp=self.create_mlp,
            #pe_emb=cfg.pe_dims
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.device = device

        #gin = GIN(cfg.n_phi_layers, 1, self.cfg.hidden_phi_layers, self.cfg.pe_dims, self.create_mlp, bn=cfg.mlp_use_bn)  # 1=eigenvec
        gin = GIN(8, 1, 120, 4, self.create_mlp, bn=True)  
        # rho = create_mlp(cfg.pe_dims * cfg.phi_hidden_dims, cfg.pe_dims)
        #rho = MLP2(cfg.n_mlp_layers, self.cfg.pe_dims, cfg.hidden_phi_layers,
        #        cfg.pe_dims, use_bn=cfg.mlp_use_bn, activation='relu', dropout_prob=0.0)
        rho = MLP2(4, 8 * 4, 120, 8, use_bn=True, activation='relu', dropout_prob=0.0)
        self.positional_encoding = SignInvPe(phi=gin, rho=rho)
        print(self.positional_encoding)
        self.pe_embedding = torch.nn.Linear(8, self.cfg.node_emb_dims)
        
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None) -> MLP:
        print(in_dims)
        return MLP2(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        W,
        print_emb=False,
        device=None
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time

        PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        reverse_node_mapping = batch.reverse_node_mapping
        #PE = self.pe_embedding(PE)
        PE_dict = {}
        last_node_idx = -1
        for homogeneous_idx, pos_encoding in enumerate(self.pe_embedding(PE)): # try with PE embedding
            node_type, node_idx = reverse_node_mapping[homogeneous_idx]
            if node_type not in PE_dict:
                last_node_idx = -1
                #print(node_type)
                PE_dict[node_type] = pos_encoding.unsqueeze(dim=0)
            else:
                PE_dict[node_type] = torch.cat((PE_dict[node_type], pos_encoding.unsqueeze(dim=0)), dim=0)
            if node_idx != last_node_idx +1 :
                print("WRONG")
                print(node_idx)
            last_node_idx = node_idx
        x_dict = self.encoder(batch.tf_dict, PE_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict, 
            #batch.reverse_node_mapping,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict
        ) # we also add pe here to each layer! SHOULD ALSO ADD PE ENCODER?

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        reverse_node_mapping = batch.reverse_node_mapping
        #PE = self.pe_embedding(PE)
        PE_dict = {}
        last_node_idx = -1
        for homogeneous_idx, pos_encoding in enumerate(self.pe_embedding(PE)): # try with PE embedding
            node_type, node_idx = reverse_node_mapping[homogeneous_idx]
            if node_type not in PE_dict:
                last_node_idx = -1
                #print(node_type)
                PE_dict[node_type] = pos_encoding.unsqueeze(dim=0)
            else:
                PE_dict[node_type] = torch.cat((PE_dict[node_type], pos_encoding.unsqueeze(dim=0)), dim=0)
            if node_idx != last_node_idx +1 :
                print("WRONG")
                print(node_idx)
            last_node_idx = node_idx
        x_dict = self.encoder(batch.tf_dict, PE_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])