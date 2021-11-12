import time
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from common.src.models.base import GWM, AtomConv, AnchorConv, UpwardConv, scatter_softmax
from common.src.models.utils import ATOM_FDIM
from common.src.models.loss import DistanceLoss

MODEL_PARAMS = {
    'GNNvert_depth': 4,
    'GNNsite_depth': 3,
    'GNNmasf_depth': 3,
    'GNNanch_depth': 0,
    'k_head': 1,
    'hidden_comp': 128,
    'hidden_prot': 128,
    'hidden_site': 64,
    'hidden_masf': 64,
    'hidden_vert': 128,
    'hidden_dis': 64,
    'site_input_dim': 131,
    'masf_input_dim': 5,
    'pairwise_type': 'all', # distance, interaction, all
    'Affsrc': 'all', # atom, masif, all

}


TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 32,
    'list_task': ['DistanceCP', 'DistanceCC', 'DistancePP'],
    'task_eval': {'DistanceCP': "reg", 'DistanceCC': "reg", 'DistancePP': "reg"},
    'task': 'DistanceCP',
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
        self.GNNmasf_depth = int(params['GNNmasf_depth'])
        self.GNNanch_depth = int(params['GNNanch_depth'])
        self.k_head = int(params['k_head'])
        self.hidden_vert = int(params['hidden_vert'])
        self.hidden_comp = int(params['hidden_comp'])
        self.hidden_site = int(params['hidden_site'])
        self.hidden_masf = int(params['hidden_masf'])
        self.hidden_prot = int(params['hidden_prot'])
        self.hidden_dis = int(params['hidden_dis'])
        self.site_input_dim = int(params['site_input_dim'])
        self.masf_input_dim = int(params['masf_input_dim'])
        self.pairwise_type = params['pairwise_type']
        self.Affsrc = params['Affsrc']
        
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
        self.AMMMasf = nn.ModuleList()
        for i in range(self.GNNsite_depth):
            if i == 0:
                self.AMMAtom.append(AtomConv(self.site_input_dim, self.hidden_site))
            else:
                self.AMMAtom.append(AtomConv(self.hidden_site, self.hidden_site))
        for i in range(self.GNNmasf_depth):
            if i == 0:
                self.AMMMasf.append(AtomConv(self.masf_input_dim, self.hidden_masf))
            else:
                self.AMMMasf.append(AtomConv(self.hidden_masf, self.hidden_masf))

        self.AMMAtom_output = nn.Sequential(
            nn.Linear(self.site_input_dim+self.GNNsite_depth*self.hidden_site, self.hidden_prot),
            nn.LeakyReLU(0.1),
        )
        self.AMMMasf_output = nn.Sequential(
            nn.Linear(self.masf_input_dim+self.GNNmasf_depth*self.hidden_masf, self.hidden_prot),
            nn.LeakyReLU(0.1),
        )

        # Anchor Aggregate Module parameters
        self.AAM = nn.ModuleDict()
        self.AAM["Atom"] = UpwardConv() 
        self.AAM["Masf"] = UpwardConv() 
        # Group Aggregate Module parameters
        self.GAM = UpwardConv() 
        # Anchor MPNN
        self.AMPNN = nn.ModuleList()
        for i in range(self.GNNanch_depth):
            if i == 0:
                if self.Affsrc == "all":
                    self.AMPNN.append(AnchorConv(self.hidden_prot*2, self.hidden_prot))
                else:
                    self.AMPNN.append(AnchorConv(self.hidden_prot, self.hidden_prot))

            else:
                self.AMPNN.append(AnchorConv(self.hidden_prot, self.hidden_prot))

        if self.Affsrc == "all":
            self.AMPNN_output = nn.Sequential(
                nn.Linear(self.hidden_prot*2+self.GNNanch_depth*self.hidden_prot, self.hidden_prot),
                nn.LeakyReLU(0.1),
            )
        else:
            self.AMPNN_output = nn.Sequential(
                nn.Linear(self.hidden_prot+self.GNNanch_depth*self.hidden_prot, self.hidden_prot),
                nn.LeakyReLU(0.1),
            )

        # Pairwise distance matrix prediction parameters
        self.distance_comp = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_dis),
            nn.LeakyReLU(0.1),
        )
        self.distance_site = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_dis),
            nn.LeakyReLU(0.1),
        )

        self.use_int, self.use_dis = False, False
        if self.pairwise_type == "interaction" or self.pairwise_type == "all":
            self.use_int = True
        if self.pairwise_type == "distance" or self.pairwise_type == "all":
            self.use_dis = True
         
           
    def load_optimizer(self, train_params):

        self.loss = {
            'DistanceCP': DistanceLoss(4), 
            'DistanceCC': DistanceLoss(4), 
            'DistancePP': DistanceLoss(4), 
            }
        self.task_eval = train_params['task_eval']

    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
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
    
    def MasfConv_module(self, masfGraphBatch):
        
        masf_feature = masfGraphBatch.x
        edge_index = masfGraphBatch.edge_index

        masf_feature_list = [masf_feature]
        for AMM_iter in range(self.GNNmasf_depth):
            masf_feature = self.AMMMasf[AMM_iter](x=masf_feature, edge_index=edge_index)
            masf_feature_list.append(masf_feature)
        
        masf_feature_output = torch.cat(masf_feature_list, dim=-1)
        masf_feature_output = self.AMMMasf_output(masf_feature_output)

        return masf_feature_output
    
    def fill_anchor_graph(self, site_feature, protGraphBatch):
        prot_feature = torch.zeros(protGraphBatch.num_nodes, site_feature.shape[1], device=site_feature.device)
        prot_feature[protGraphBatch.mask] = site_feature

        return prot_feature  

    def Group_module(self, compGraphBatch):
        group_feature = compGraphBatch.x
        edge_index = compGraphBatch.edge_index
        _, edge_batch = torch.unique(edge_index[1], return_inverse=True)
        edge_weight = scatter_softmax(torch.ones((edge_index.shape[1], 1), device=group_feature.device), edge_batch)

        group_feature = self.GAM(x=group_feature, edge_index=edge_index, edge_weight=edge_weight)

        return group_feature[compGraphBatch.mask == False]

    def Anchor_module(self, protGraphBatch, feature_type):
        anchor_feature = protGraphBatch.x
        edge_index = protGraphBatch.edge_index
        edge_attr = protGraphBatch.edge_attr
        if edge_index.shape[1] == 0:
            edge_weight = edge_attr
        else:
            _, edge_batch = torch.unique(edge_index[1], return_inverse=True)
            edge_weight = scatter_softmax(6-edge_attr, edge_batch)
        
        anchor_feature = self.AAM[feature_type](x=anchor_feature, edge_index=edge_index, edge_weight=edge_weight)

        return anchor_feature[protGraphBatch.mask == False]

    def AnchorAnchor_module(self, anchGraphBatch):
        anch_feature = anchGraphBatch.x
        edge_index = anchGraphBatch.edge_index

        anch_feature_list = [anch_feature]
        for AMPNN_iter in range(self.GNNanch_depth):
            anch_feature = self.AMPNN[AMPNN_iter](x=anch_feature, edge_index=edge_index)
            anch_feature_list.append(anch_feature)
        
        anch_feature_output = torch.cat(anch_feature_list, dim=-1)
        anch_feature_output = self.AMPNN_output(anch_feature_output)

        return anch_feature_output  

    def get_distance_index(self, protGraphBatch, compGraphBatch):
        index_ab, index_aa, index_bb = [], [], []
        total_anchor, total_vertex = 0, 0
        device = protGraphBatch.edge_index.device
        for anchor, group in zip(protGraphBatch.kwargs, compGraphBatch.kwargs):
            n_anchor = anchor['num_anchor']
            n_vertex = group['num_anchor']
            idx_a, idx_b = torch.meshgrid(torch.arange(n_anchor, device=device), torch.arange(n_vertex, device=device))
            index_ab.append(torch.cat([idx_a.reshape(-1, 1) + total_anchor, idx_b.reshape(-1, 1) + total_vertex], dim=1))
            index_aa.append(torch.triu_indices(n_anchor, n_anchor, 1, device=device).T + total_anchor)
            index_bb.append(torch.triu_indices(n_vertex, n_vertex, 1, device=device).T + total_vertex)
            total_anchor += n_anchor
            total_vertex += n_vertex
        index_ab = torch.cat(index_ab, dim=0)
        index_aa = torch.cat(index_aa, dim=0)
        index_bb = torch.cat(index_bb, dim=0)

        return index_ab, index_aa, index_bb
    
    def Distance_pred_module(self, anchor_feature, group_feature, index_ag, index_aa, index_gg):
        pairwise_anchor_feature = self.distance_site(anchor_feature)
        pairwise_group_feature = self.distance_comp(group_feature)
        
        ag_distance_pred = torch.pairwise_distance(pairwise_anchor_feature.index_select(0, index_ag[:,0]), 
                                                   pairwise_group_feature.index_select(0, index_ag[:,1])).view(-1, 1)
        aa_distance_pred = torch.pairwise_distance(pairwise_anchor_feature.index_select(0, index_aa[:,0]), 
                                                   pairwise_anchor_feature.index_select(0, index_aa[:,1])).view(-1, 1)
        gg_distance_pred = torch.pairwise_distance(pairwise_group_feature.index_select(0, index_gg[:,0]), 
                                                   pairwise_group_feature.index_select(0, index_gg[:,1])).view(-1, 1)

        return ag_distance_pred, aa_distance_pred, gg_distance_pred

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
        grou_feature = self.Group_module(compGraphBatch)

        # Step 3. Pocket GNN
        """
        Graph message passing on proteins
        site_feature: protein atoms
        masf_feature: points sampled from protein surface with features named MaSIF
        """
        # if calcTime: list_time.append(time.time())
        site_feature = self.AtomConv_module(siteGraphBatch)
        masf_feature = self.MasfConv_module(masfGraphBatch)
        
        # Step 4. Anchor GNN
        """
        site  -> anchor
        masif -> anchor
        what is anchor? see below
        """
        # if calcTime: list_time.append(time.time())
        protGraphBatch.x = self.fill_anchor_graph(site_feature, protGraphBatch)
        promGraphBatch.x = self.fill_anchor_graph(masf_feature, promGraphBatch)

        if calcTime: list_time.append(time.time())
        """
        anchor is points sampled from the cavity space of protein pockets
        protein anchors are to be in accordance with compound groups
        """
        anch_atom_feature = self.Anchor_module(protGraphBatch, "Atom")
        anch_masf_feature = self.Anchor_module(promGraphBatch, "Masf")
        if self.Affsrc == "all":
            anchGraphBatch.x = torch.cat([anch_atom_feature, anch_masf_feature], dim=-1)
        elif self.Affsrc == "atom":
            anchGraphBatch.x = anch_atom_feature
        elif self.Affsrc == "masif":
            anchGraphBatch.x = anch_masf_feature
        else:
            raise NotImplementedError

        anch_feature = self.AnchorAnchor_module(anchGraphBatch)

        # Step 6. Pairwise distance prediction
        """
        predict pairwise distance between group and anchor
        distance in 3d space
        """
        if calcTime: list_time.append(time.time())
        index_ag, index_aa, index_gg = self.get_distance_index(protGraphBatch, compGraphBatch)
        distance_pred, pp_distance_pred, cc_distance_pred = self.Distance_pred_module(anch_feature, grou_feature, index_ag, index_aa, index_gg)
        
        # Step 8. Output
        if calcTime: list_time.append(time.time())
        if calcTime: 
            list_time = np.array(list_time)
            time_elapsed = list_time[1:]-list_time[:-1]
            print(("{:.4f}\t"*(len(list_time)-1)).format(*time_elapsed))

        return {
            "DistanceCP": distance_pred, 
            "DistanceCC": cc_distance_pred, 
            "DistancePP": pp_distance_pred, 
            }
