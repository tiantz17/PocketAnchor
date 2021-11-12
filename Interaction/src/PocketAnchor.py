import time
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from common.src.models.base import GWM, AtomConv
from common.src.models.utils import ATOM_FDIM

MODEL_PARAMS = {
    'GNNvert_depth': 4,
    'GNNsite_depth': 3,
    'k_head': 1,
    'hidden_comp': 128,
    'hidden_prot': 128,
    'hidden_site': 64,
    'hidden_vert': 128,
    'hidden_int': 64,
    'site_input_dim': 131,
}


TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 32,
    'list_task': ['Interaction'],
    'loss_weight': {'Interaction': 1.0},
    'task_eval': {'Interaction': "cls"},
    'task': 'Interaction',
}


class Model(nn.Module):
    """
    PocketAnchor for affinity prediction
    """
    def __init__(self, params):
        super(Model, self).__init__()

        """hyper part"""
        self.GNNvert_depth = int(params['GNNvert_depth'])
        self.GNNsite_depth = int(params['GNNsite_depth'])
        self.k_head = int(params['k_head'])
        self.hidden_vert = int(params['hidden_vert'])
        self.hidden_comp = int(params['hidden_comp'])
        self.hidden_site = int(params['hidden_site'])
        self.hidden_prot = int(params['hidden_prot'])
        self.hidden_int = int(params['hidden_int'])
        
        """GraphConv Module"""
        # First transform vertex features into hidden representations
        self.vert_embedding = nn.Sequential(  
            nn.Linear(ATOM_FDIM, self.hidden_vert),
            nn.LeakyReLU(0.1),
        )
        # Graph Warp Module parameters
        self.GRU_main = nn.GRUCell(self.hidden_vert, self.hidden_vert)
        self.GRU_super = nn.GRUCell(self.hidden_vert, self.hidden_vert)
        self.GWM = nn.ModuleList([GWM(self.hidden_vert, self.GRU_main, self.GRU_super, self.k_head) for _ in range(self.GNNvert_depth)])
        
        # Atom/Masif Module parameters
        self.AMMAtom = nn.ModuleList()
        for i in range(self.GNNsite_depth):
            if i == 0:
                self.AMMAtom.append(AtomConv(self.site_input_dim, self.hidden_site))
            else:
                self.AMMAtom.append(AtomConv(self.hidden_site, self.hidden_site))

        self.AMMAtom_output = nn.Sequential(
            nn.Linear(self.site_input_dim+self.GNNsite_depth*self.hidden_site, self.hidden_prot),
            nn.LeakyReLU(0.1),
        )

        self.num_interaction = 7
        # Pairwise interaction matrix prediction parameters
        self.interaction_comp = nn.ModuleList([nn.Sequential(
                nn.Linear(self.hidden_vert, self.hidden_int),
                nn.LeakyReLU(0.1),
            ) for _ in range(self.num_interaction)]
        )
        self.interaction_site = nn.ModuleList([nn.Sequential(
                nn.Linear(self.hidden_prot, self.hidden_int),
                nn.LeakyReLU(0.1),
            ) for _ in range(self.num_interaction)]
        )
            
           
    def load_optimizer(self, train_params):

        self.loss = {
            'Interaction': nn.BCEWithLogitsLoss(reduction='none')
            }
        self.task_eval = train_params['task_eval']


    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
            if task == "Interaction":
                interact_weight = 1 * (dict_label[task] == 0) + dict_label[task] * (dict_label[task] > 0)
                loss = loss + torch.mean(self.loss[task](dict_pred[task], dict_label[task]) * interact_weight)
            else:
                loss = loss + self.loss[task](dict_pred[task], dict_label[task]) * self.loss_weight[task]
        return loss

    def GraphConv_module(self, vertGraphBatch):
        vert_initial = vertGraphBatch.x

        # vertex and group embedding
        vert_feature = self.vert_embedding(vert_initial)

        supe_feature = scatter_sum(vert_feature, vertGraphBatch.batch, dim=0)
        
        for GWM_iter in range(self.GNNvert_depth):
            vert_feature, supe_feature = self.GWM[GWM_iter](vert_feature, supe_feature, vertGraphBatch)

        return vert_feature, supe_feature 

    def AtomConv_module(self, siteGraphBatch):
        site_feature = siteGraphBatch.x
        edge_index = siteGraphBatch.edge_index

        site_feature_list = [site_feature]
        for AMM_iter in range(self.GNNsite_depth):
            site_feature = self.AMMAtom[AMM_iter](x=site_feature, edge_index=edge_index)
            site_feature_list.append(site_feature)
        
        site_feature_output = torch.cat(site_feature_list, dim=-1)
        site_feature_output = self.AMMAtom_output(site_feature_output)

        return site_feature_output
    
    def fill_anchor_graph(self, site_feature, protGraphBatch):
        prot_feature = torch.zeros(protGraphBatch.num_nodes, site_feature.shape[1], device=site_feature.device)
        prot_feature[protGraphBatch.mask] = site_feature

        return prot_feature  

    def get_interaction_index(self, siteGraphBatch, vertGraphBatch):
        index_tc = []
        total_site, total_comp = 0, 0
        device = siteGraphBatch.x.device
        for pocket, molecule in zip(siteGraphBatch.to_data_list(), vertGraphBatch.contiguous().to_data_list()):
            n_site = pocket.num_nodes
            n_comp = molecule.num_nodes
            idx_a, idx_g = torch.meshgrid(torch.arange(n_site, device=device), torch.arange(n_comp, device=device))
            index_tc.append(torch.cat([idx_a.reshape(-1, 1) + total_site, idx_g.reshape(-1, 1) + total_comp], dim=1))
            total_site += n_site
            total_comp += n_comp
        index_tc = torch.cat(index_tc, dim=0)

        return index_tc

    def Interaction_pred_module(self, site_feature, vertex_feature, index_ag):
        interact_pred = []
        for idx in range(self.num_interaction):
            pairwise_a_feature = self.interaction_site[idx](site_feature)
            pairwise_g_feature = self.interaction_comp[idx](vertex_feature)
            interact_pred.append(torch.sum(pairwise_a_feature.index_select(0, index_ag[:,0]) * 
                pairwise_g_feature.index_select(0, index_ag[:,1]), dim=1, keepdim=True))
        
        interact_pred = torch.cat(interact_pred, dim=1)
        return interact_pred

    def forward(self, vertGraphBatch, grouGraphBatch, compGraphBatch, siteGraphBatch, masfGraphBatch, anchGraphBatch, protGraphBatch, promGraphBatch):
        calcTime = False
        list_time = []
        device = vertGraphBatch.x.device

        # Step 1. Compound GNN        
        """
        Graph message passing on compounds
        Graph warp module, with pseudo node for global representation.
        vert_features: compound atom (vertex)
        super_feature: compound global representation (super node)
        """
        if calcTime: list_time.append(time.time())
        vert_feature, super_feature = self.GraphConv_module(vertGraphBatch)

        # # Step 2. Group GNN
        """
        GNN upon functional groups of compounds
        compGraphBatch is a special implementation for message passing from vert (compound atoms) to grou (functional groups)
        It contains nodes of verts and groups, and edges from verts to groups
        self.fill_anchor_graph() aims to put vert features into this special graph and get the group features
        """
        if calcTime: list_time.append(time.time())
        compGraphBatch.x = self.fill_anchor_graph(vert_feature, compGraphBatch)

        # Step 3. Pocket GNN
        """
        Graph message passing on proteins
        site_feature: protein atoms
        masf_feature: points sampled from protein surface with features named MaSIF
        """
        # if calcTime: list_time.append(time.time())
        site_feature = self.AtomConv_module(siteGraphBatch)
        
        # Step 5. Pairwise interaction prediction
        """
        predict pariwise interaction between vert and site (atom-wise interaction)
        normally non-valence interaction 
        """
        if calcTime: list_time.append(time.time())
        # SLOW
        index_tc = self.get_interaction_index(siteGraphBatch, vertGraphBatch)
        interact_pred = self.Interaction_pred_module(site_feature, vert_feature, index_tc)
                
        # Step 8. Output
        if calcTime: list_time.append(time.time())
        if calcTime: 
            list_time = np.array(list_time)
            time_elapsed = list_time[1:]-list_time[:-1]
            print(("{:.4f}\t"*(len(list_time)-1)).format(*time_elapsed))

        return {
            "Interaction": interact_pred
            }
