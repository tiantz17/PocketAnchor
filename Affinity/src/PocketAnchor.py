import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_sum

from common.src.models.base import GWM, AtomConv, AnchorConv, UpwardConv, scatter_softmax
from common.src.models.utils import ATOM_FDIM

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
    'hidden_aff': 128, 
    'site_input_dim': 131,
    'masf_input_dim': 5,
    'pairwise_type': 'all', # distance, interaction, all
    'Affsrc': 'all', # atom, masif, all
}


TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 32,
    'list_task': ['Affinity', ],
    'task_eval': {'Affinity': "reg", },
    'task': 'Affinity',
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
        self.hidden_aff = int(params['hidden_aff'])
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

        self.use_int, self.use_dis = False, False
        if self.pairwise_type == "interaction" or self.pairwise_type == "all":
            self.use_int = True
        if self.pairwise_type == "distance" or self.pairwise_type == "all":
            self.use_dis = True
            
        if self.use_int:
            # affinity prediction parameters
            self.c_aff_int = nn.Sequential(
                nn.Linear(self.hidden_vert, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )
            self.p_aff_int = nn.Sequential(
                nn.Linear(self.hidden_prot, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )

        if self.use_dis:
            # affinity prediction parameters
            self.c_aff_dis = nn.Sequential(
                nn.Linear(self.hidden_comp, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )
            self.p_aff_dis = nn.Sequential(
                nn.Linear(self.hidden_prot, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )

        if self.use_dis:
            self.att_comp = nn.Linear(self.hidden_aff, 1)
            self.att_prot = nn.Linear(self.hidden_aff, 1)
        if self.use_int:
            self.att_vert = nn.Linear(self.hidden_aff, 1)
            self.att_site = nn.Linear(self.hidden_aff, 1)
     
        if self.use_int:
            self.super_aff_int = nn.Sequential(
                nn.Linear(self.hidden_vert, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )
        if self.use_dis:
            self.super_aff_dis = nn.Sequential(
                nn.Linear(self.hidden_comp, self.hidden_aff),
                nn.LeakyReLU(0.1),
            )
        if self.use_int and self.use_dis:
            self.W_out = nn.Linear(self.hidden_aff*self.hidden_aff*2*2, 1)              
        else:
            self.W_out = nn.Linear(self.hidden_aff*self.hidden_aff*2, 1)
                
           
    def load_optimizer(self, train_params):
        self.loss = {
            'Affinity': nn.MSELoss(), 
            }
        self.task_eval = train_params['task_eval']

    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
            if self.loss_weight[task] == 0.0:
                continue
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

    def Affinity_pred_module(self, site_feature, prot_feature, vert_feature, comp_feature, super_feature, \
        batch_site, batch_prot, batch_vert, batch_comp):
        """
        projecting into the same space
        """
        if self.use_dis:
            comp_embedding = self.c_aff_dis(comp_feature)
            prot_embedding = self.p_aff_dis(prot_feature)
        if self.use_int:
            vert_embedding = self.c_aff_int(vert_feature)
            site_embedding = self.p_aff_int(site_feature)

        """
        Attention 
        """        
        if self.use_dis:
            comp_attention = scatter_softmax(self.att_comp(comp_embedding), batch_comp)
            prot_attention = scatter_softmax(self.att_prot(prot_embedding), batch_prot)
            comp_feature = scatter_sum(comp_attention * comp_embedding, batch_comp, dim=0)
            prot_feature = scatter_sum(prot_attention * prot_embedding, batch_prot, dim=0)
        if self.use_int:
            vert_attention = scatter_softmax(self.att_vert(vert_embedding), batch_vert)
            site_attention = scatter_softmax(self.att_site(site_embedding), batch_site)
            vert_feature = scatter_sum(vert_attention * vert_embedding, batch_vert, dim=0)
            site_feature = scatter_sum(site_attention * site_embedding, batch_site, dim=0)
        

        """
        Using kroneck product for prediction
        """
        if self.use_dis:
            super_feature_dis = self.super_aff_dis(super_feature)
            comp_feature = torch.cat([comp_feature, super_feature_dis], dim=1)
            kroneck_dis = torch.matmul(comp_feature.unsqueeze(2), prot_feature.unsqueeze(1))
            kroneck_dis = F.leaky_relu(kroneck_dis.view(kroneck_dis.shape[0],-1), 0.1)

        if self.use_int:
            super_feature_int = self.super_aff_int(super_feature)
            vert_feature = torch.cat([vert_feature, super_feature_int], dim=1)
            kroneck_int = torch.matmul(vert_feature.unsqueeze(2), site_feature.unsqueeze(1))
            kroneck_int = F.leaky_relu(kroneck_int.view(kroneck_int.shape[0],-1), 0.1)

        if self.use_dis and self.use_int:
            affinity_pred = self.W_out(torch.cat([kroneck_dis, kroneck_int], dim=1))
        elif self.use_dis:
            affinity_pred = self.W_out(kroneck_dis)
        elif self.use_int:
            affinity_pred = self.W_out(kroneck_int)
        return affinity_pred

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

        # Step 7. Affinity prediction
        """
        predict affinity between compound and protein
        """
        if calcTime: list_time.append(time.time())
        # SLOW
        batch_site = torch.cat([torch.LongTensor([i]*anchor['num_site']) for i, anchor in enumerate(protGraphBatch.kwargs)]).to(device)
        batch_anch = torch.cat([torch.LongTensor([i]*anchor['num_anchor']) for i, anchor in enumerate(protGraphBatch.kwargs)]).to(device)
        batch_vert = torch.cat([torch.LongTensor([i]*group['num_site']) for i, group in enumerate(compGraphBatch.kwargs)]).to(device)
        batch_grou = torch.cat([torch.LongTensor([i]*group['num_anchor']) for i, group in enumerate(compGraphBatch.kwargs)]).to(device)

        affinity_pred = self.Affinity_pred_module(site_feature, anch_feature, vert_feature, grou_feature, super_feature, \
            batch_site, batch_anch, batch_vert, batch_grou)
        
        # Step 8. Output
        if calcTime: list_time.append(time.time())
        if calcTime: 
            list_time = np.array(list_time)
            time_elapsed = list_time[1:]-list_time[:-1]
            print(("{:.4f}\t"*(len(list_time)-1)).format(*time_elapsed))

        return {
            "Affinity": affinity_pred, 
            }
