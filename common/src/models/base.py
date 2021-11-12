import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

import numpy as np

from common.src.models.utils import ATOM_FDIM, BOND_FDIM

from torch_scatter import scatter_sum, scatter_max, scatter_mean


def scatter_softmax(a, index, dim=0):
    """
    softmax for scatter data structure
    """
    a_max, _ = scatter_max(a, index, dim)
    a_exp = torch.exp(a - a_max.index_select(0, index))
    a_sum = scatter_sum(a_exp, index, dim) + 1e-6
    a_softmax = a_exp / a_sum.index_select(0, index)
    return a_softmax


class WLNConv(MessagePassing):
    """
    Weisfeiler Lehman relabelling layer
    """
    def __init__(self, in_channels, out_channels):
        super(WLNConv, self).__init__(aggr='add')
        # WLN parameters
        self.label_U2 = nn.Sequential( #assume no edge feature transformation
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.label_U1 = nn.Linear(out_channels*2, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            z = x_j
        else:
            z = torch.cat([x_j, edge_attr], dim=-1)
        return self.label_U2(z)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.label_U1(z)


class GWM(nn.Module):
    """
    Graph Warp Module, see paper for more detail
    """
    def __init__(self, hidden_comp, GRU_main, GRU_super, k_head=1):
        super(GWM, self).__init__()
        self.hidden_comp = hidden_comp
        self.k_head = k_head
        
        # Transmitter parameters
        self.W_a_main = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 
        self.W_a_super = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 
        self.W_main = nn.ModuleList([
            nn.Linear(self.hidden_comp, self.hidden_comp) for i in range(self.k_head)]) 
        self.W_bmm = nn.ModuleList([
            nn.Linear(self.hidden_comp, 1) for i in range(self.k_head)]) 
        
        self.W_super = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) 
        self.W_main_to_super = nn.Sequential(
            nn.Linear(self.hidden_comp*self.k_head, self.hidden_comp),
            nn.Tanh(),
            ) 
        self.W_super_to_main = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            )
        
        # Warp gate
        self.W_zm1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zm2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.GRU_main = GRU_main
        self.GRU_super = GRU_super
        
        # WLN parameters
        self.WLN = WLNConv(self.hidden_comp+BOND_FDIM, self.hidden_comp)
    
    def forward(self, vertex_feature, super_feature, molGraphBatch):
        edge_initial = molGraphBatch.edge_attr
        edge_index = molGraphBatch.edge_index
        batch = molGraphBatch.batch
        
        # prepare main node features
        for k in range(self.k_head):
            a_main = self.W_a_main[k](vertex_feature)
            a_super = self.W_a_super[k](super_feature)
            a = self.W_bmm[k](a_main * a_super.index_select(0, batch))
            attn = scatter_softmax(a.view(-1), batch).view(-1, 1)
            k_main_to_super = scatter_sum(attn * self.W_main[k](vertex_feature), batch, dim=0)
            if k == 0:
                m_main_to_super = k_main_to_super
            else:
                m_main_to_super = torch.cat([m_main_to_super, k_main_to_super], dim=-1)  # concat k-head
        main_to_super = self.W_main_to_super(m_main_to_super)
        super_to_main = self.W_super_to_main(super_feature)

        main_self = self.WLN(x=vertex_feature, edge_index=edge_index, edge_attr=edge_initial)  
        super_self = self.W_super(super_feature)

        # warp gate and GRU for update main node features, use main_self and super_to_main
        z_main = torch.sigmoid(self.W_zm1(main_self) + self.W_zm2(super_to_main).index_select(0, batch)) 
        hidden_main = (1-z_main)*main_self + z_main*super_to_main.index_select(0, batch)
        vertex_feature = self.GRU_main(hidden_main, vertex_feature)
        # warp gate and GRU for update super node features
        z_supper = torch.sigmoid(self.W_zs1(super_self) + self.W_zs2(main_to_super))  
        hidden_super = (1-z_supper)*super_self + z_supper*main_to_super  
        super_feature = self.GRU_super(hidden_super, super_feature)

        return vertex_feature, super_feature


class AtomConv(MessagePassing):
    """
    Message Passing for atom
    """
    def __init__(self, in_channels, out_channels):
        super(AtomConv, self).__init__(aggr='add')
        # parameters
        self.U2 = nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.U1 = nn.Linear(in_channels+out_channels, out_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_j):
        return self.U2(x_j)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.U1(z)


class AnchorConv(MessagePassing):
    """
    Message Passing for anchor
    """
    def __init__(self, in_channels, out_channels):
        super(AnchorConv, self).__init__(aggr='add')
        # parameters
        self.U2 = nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.U1 = nn.Linear(in_channels+out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return self.U2(x_j) * edge_weight
        else:
            return self.U2(x_j)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.U1(z)


class UpwardConv(MessagePassing):
    """
    Message Passing from bottom level to top level
    e.g., atom   -> anchor
          masif  -> anchor
          vertex -> group
    """
    def __init__(self):
        super(UpwardConv, self).__init__(aggr='add')
        # parameters
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, edge_weight):
        # shape of edge
        if edge_weight is not None:
            return x_j * edge_weight
        else:
            return x_j

    def update(self, message):
        # shape of node
        return message

