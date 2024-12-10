from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from relbench.modeling.gin import GIN
from relbench.modeling.mlp import MLP

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from scipy.special import comb
import math

def filter1(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK

def bern_filter(S, W, k):
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(1, k): 
        L = (1/(2**k)) * math.comb(k, i) * torch.linalg.matrix_power(
                                    (2*(torch.eye(S.shape[0]).to(S.device)) - S), k) @ S
        out = L @ W # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1)

class K_PEARL_PE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList
    def __init__(self, phi: nn.Module, BASIS, k=16, mlp_nlayers=1, mlp_hid=16, spe_act='relu', mlp_out=16) -> None:
        super().__init__()
        #out_dim = len(psi_list)
        print("In spe mlp using activation: ", spe_act)
        self.mlp_nlayers = mlp_nlayers
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert(mlp_hid == mlp_out)
            self.bn = nn.ModuleList()
            self.mlp_nlayers = mlp_nlayers
            self.layers = nn.ModuleList([nn.Linear(k if i==0 else mlp_hid, 
                                        mlp_hid if i<mlp_nlayers-1 else mlp_out, bias=True) for i in range(mlp_nlayers)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(mlp_hid if i<mlp_nlayers-1 else mlp_out,track_running_stats=True) for i in range(mlp_nlayers)])
        if spe_act == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif spe_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = SwiGLU(mlp_hid) ## edit if you want more than 1 mlp layers!!
        self.phi = phi
        self.k = k
        self.BASIS = BASIS
        self.running_sum = 0
        self.total = 0
        print("SPE BASIS IS: ", self.BASIS)
        print("SPE k is: ", self.k)

    def forward(
        self, Lap, W, edge_index: torch.Tensor, final=False
    ) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        # for loop N times for each Nx1 e
        for lap, w in zip(Lap, W):
            output = filter1(lap, w, self.k)#bern_filter(lap, w, self.k)
            if self.mlp_nlayers > 0:
                for layer, bn in zip(self.layers, self.norms):
                    output = output.transpose(0, 1)
                    output = layer(output)
                    output = bn(output.transpose(1,2)).transpose(1,2)
                    output = self.activation(output)
                    output = output.transpose(0, 1)
            W_list.append(output)             # [NxMxK]*B
        return self.phi(W_list, edge_index, self.BASIS, final=final)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims


class PEARL_PE1(nn.Module):
    # pe = rho(phi(V)+phi(-V))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(PEARL_PE1, self).__init__()
        self.phi = phi
        self.rho = rho
        self.running_sum = 0
        self.total = 0

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor,
            accumulate, final
    ) -> torch.Tensor:
        x = V #NxMx1
        x = self.phi(x, edge_index) 
        if accumulate:
            self.running_sum += x.sum(dim=1) #+ self.phi(-x, edge_index) # [N, D_pe, hidden_dims]
            self.total += x.shape[1]
        else:
            x = x.mean(dim=1)
            x = self.rho(x) 
        if accumulate and final:
            x = self.running_sum / self.total
            x = self.rho(x) 
            self.running_sum = 0
            self.total = 0

        return x

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims


    
class MaskedSignInvPe(nn.Module):
    # pe = rho(mask-sum(phi(V)+phi(-V)))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(MaskedSignInvPe, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1)  # TO DO: incorporate eigenvalues
        x = self.phi(x, edge_index) + self.phi(-x, edge_index)  # [N, D_pe, hidden_dims]
        pe_dim, N = x.size(1), x.size(0)
        num_nodes = [torch.sum(batch == i) for i in range(batch[-1]+1)]
        a = torch.arange(0, pe_dim).to(x.device)
        mask = torch.cat([(a < num).unsqueeze(0).repeat([num, 1]) for num in num_nodes], dim=0) # -1 since excluding zero eigenvalue
        x = (x*mask.unsqueeze(-1)).sum(dim=1) # [N, hidden_dims]
        x = self.rho(x)  # [N, D_pe]
        return x


class SignInvPe(nn.Module):
    # pe = rho(phi(V)+phi(-V))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(SignInvPe, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1) # TO DO: incorporate eigenvalues
        x = self.phi(x, edge_index) + self.phi(-x, edge_index) # [N, D_pe, hidden_dims]
        x = x.reshape([x.shape[0], -1]) # [N, D_pe * hidden_dims]
        x = self.rho(x) # [N, D_pe]

        return x

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims



class MLPPhi(nn.Module):
    gin: GIN

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()
        # self.mlp = MLP(n_layers, in_dims, hidden_dims, out_dims, use_bn=False, activation='relu', dropout_prob=0.0)
        test_mlp = create_mlp(1, 1)
        use_bn, dropout_prob = test_mlp.layers[0].bn is not None, test_mlp.dropout.p
        self.mlp = MLP(n_layers, in_dims, hidden_dims, out_dims, use_bn=use_bn, activation='relu',
                       dropout_prob=dropout_prob, norm_type="layer")
        del test_mlp

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
        PE = self.mlp(W)       # [N_sum, N_max, D_pe]
        return (PE * mask.unsqueeze(-1)).sum(dim=1)               # [N_sum, D_pe]
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims



class GINPhi(nn.Module):
    gin: GIN

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool, RAND_LAP, pooling=True
    ) -> None:
        super().__init__()
        print('pooling is,', pooling)
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn, laplacian=RAND_LAP)
        #self.mlp = create_mlp(out_dims, out_dims, use_bias=True)
        self.running_sum = 0
        self.total = 0
        self.pooling=pooling

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, BASIS, mean=False, running_sum=True, final=False) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """ 
        if not BASIS:
            W = torch.cat(W_list, dim=0)   # [N_sum, M, K]
            PE = self.gin(W, edge_index)  # [N,M,D]
            if mean:
                PE = (PE).mean(dim=1) # sum or mean along M? get N, D_pe
            else:
                if running_sum:
                    if self.pooling:
                        self.running_sum += (PE).sum(dim=1)
                    else:
                        self.running_sum += PE
                PE = PE
                if final:
                    PE = self.running_sum
                    self.running_sum = 0
                #if self.pooling:
                #    PE = (PE).sum(dim=1)
            return PE               # [N_sum, D_pe]
        else:
            W = W_list[0]
            PE = self.gin(W, edge_index)
            if running_sum:
                self.running_sum += (PE).sum(dim=1)
            if final:
                PE = self.running_sum
                self.running_sum = 0
            return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims

def GetPhi(cfg, create_mlp: Callable[[int, int], MLP], device):
    return GINPhi(cfg.n_phi_layers, cfg.RAND_mlp_out, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm, RAND_LAP=cfg.RAND_LAP)
