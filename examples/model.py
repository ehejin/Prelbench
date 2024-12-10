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
from relbench.modeling.pe import K_PEARL_PE, GINPhi, PEARL_PE1, MaskedSignInvPe, SignInvPe
from relbench.modeling.gin import GIN
import numpy as np

class Model(torch.nn.Module):

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
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
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
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
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
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
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
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
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
        )

        return self.head(x_dict[dst_table])
    

class Model_PEARL(torch.nn.Module):

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
        self.PE1 = PE1
        self.num_samples = cfg.num_samples
        self.REL = REL
        if PE1:
            gin = GIN(cfg.n_phi_layers, 1, self.cfg.hidden_phi_layers, self.cfg.pe_dims, self.create_mlp, bn=cfg.mlp_use_bn)  # 1=eigenvec
            # rho = create_mlp(cfg.pe_dims * cfg.phi_hidden_dims, cfg.pe_dims)
            rho = MLP2(cfg.n_mlp_layers, self.cfg.pe_dims, cfg.hidden_phi_layers,
                    cfg.pe_dims, use_bn=cfg.mlp_use_bn, activation='relu', dropout_prob=0.0)
            self.positional_encoding = PEARL_PE1(phi=gin, rho=rho)
            print(self.positional_encoding)
            self.pe_embedding = torch.nn.Linear(self.positional_encoding.out_dims, self.cfg.node_emb_dims)
        else:
            if not self.REL:
                Phi = GINPhi(self.cfg.n_phi_layers, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn, RAND_LAP=False)
                    
                self.positional_encoding = K_PEARL_PE(Phi, cfg.BASIS, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, 
                            mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)

                self.pe_embedding = torch.nn.Linear(self.positional_encoding.out_dims, self.cfg.node_emb_dims)
            else:
                self.pe_embedding = {}
                self.positional_encoding = {}
                for rel_type in ['f2p_customer_id', 'rev_f2p_customer_id', 'f2p_article_id', 'rev_f2p_article_id']:
                    #['f2p_raceId', 'rev_f2p_raceId', 'f2p_constructorId', 'rev_f2p_constructorId', 'f2p_driverId', 'rev_f2p_driverId', 'f2p_circuitId', 'rev_f2p_circuitId']: 
                    Phi = GINPhi(self.cfg.n_phi_layers, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn, RAND_LAP=False)
                    
                    self.positional_encoding[rel_type] = K_PEARL_PE(Phi, cfg.BASIS, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, 
                                mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)

                    self.pe_embedding[rel_type] = torch.nn.Linear(self.positional_encoding[rel_type].out_dims, self.cfg.node_emb_dims)

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

        if self.cfg.BASIS:
            N = batch.Lap[0].shape[0] 
            k = self.cfg.splits  
            chunk_size = N // k
            for chunk_start in range(0, N, chunk_size):
                W_list = []
                chunk_end = min(chunk_start + chunk_size, N) 
                W_chunk = torch.eye(N)[chunk_start:chunk_end].to(self.device)
                W_list.append(W_chunk)
                is_last_chunk = (idx == k - 1) or (chunk_end == N)
                if not is_last_chunk:
                    with torch.no_grad():
                        self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
                else:
                    PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
                del W_list, W
                torch.cuda.empty_cache()
        else: 
            splits = cfg.splits
            for k in range(splits):
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.cfg.BASIS:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        W = 1+torch.randn(batch.Lap[i].shape[0],self.num_samples//splits).to(self.device) #BxNxM
                    W_list.append(W)
                if k < splits-1:
                    with torch.no_grad():
                        self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
                else:
                    PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
                del W_list, W
                torch.cuda.empty_cache()
            if print_emb:
                print(PE.shape)
                np.save('./embeddings5.npy', PE.detach().cpu().numpy())
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

        '''if self.PE1:
            for i in range(19):
                r = 1+torch.randn(batch.num_nodes, self.num_samples // 20, 1).to(self.device) # NxMx[[1
                self.positional_encoding.forward(None, r, batch.edge_index, accumulate=True, final=False)
            r = 1+torch.randn(batch.num_nodes, self.num_samples // 20, 1).to(self.device)
            PE = self.positional_encoding(None, r, batch.edge_index, accumulate=True, final=True)
        else:
            # W is BxNxM
            for k in range(1):
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.cfg.BASIS:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        W = torch.randn(batch.Lap[i].shape[0],self.num_samples//4).to(self.device) #BxNxM
                    if len(W.shape) < 2:
                        print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
                    W_list.append(W)
                self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index)
            if self.cfg.BASIS:
                W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
            else:
                W = torch.randn(batch.Lap[i].shape[0],self.num_samples//4).to(self.device) #BxNxM
            if len(W.shape) < 2:
                print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)
            PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
            if print_embs:
                print(PE.shape)
                np.save('./embeddings4.npy', PE.detach().cpu().numpy())'''
                #torch.save(PE.detach().cpu().numpy(), './embeddings2.npy')
        #x_dict = x_dict + self.pe_embedding(PE)
        #x_dict = {key: x_dict[key]+self.pe_embedding(PE) for key in x_dict.keys() if x_dict[key].shape[0] > 0}
        
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

        if self.cfg.BASIS:
            k = self.cfg.splits  
            N = batch.Lap[0].shape[0] 
            chunk_size = N // k
            for idx, chunk_start in enumerate(range(0, N, chunk_size)):
                W_list = []
                chunk_end = min(chunk_start + chunk_size, N) 
                W_chunk = torch.eye(N)[:, chunk_start:chunk_end].to(self.device)
                W_list.append(W_chunk)
                is_last_chunk = (idx == k - 1) or (chunk_end == N)
                if not is_last_chunk:
                    with torch.no_grad():
                        self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
                else:
                    PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
                del W_list
                torch.cuda.empty_cache()
        else:
            splits = self.cfg.splits 
            for k in range(splits):
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.cfg.BASIS:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        W = 1+torch.randn(batch.Lap[i].shape[0],self.num_samples//splits).to(self.device) #BxNxM
                    W_list.append(W)
                if k < splits-1:
                    with torch.no_grad():
                        self.positional_encoding.forward(batch.Lap, W_list, batch.edge_index, final=False)
                else:
                    PE = self.positional_encoding(batch.Lap, W_list, batch.edge_index, final=True)
                del W_list, W
                torch.cuda.empty_cache()
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