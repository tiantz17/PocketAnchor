import time
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.src.models.base import AtomConv, AnchorConv, UpwardConv, scatter_softmax


MODEL_PARAMS = {
    'GNNsite_depth': 2,
    'GNNmasf_depth': 2,
    'GNNanch_depth': 1,
    'hidden_anch': 128,
    'hidden_site': 64,
    'hidden_masf': 64,
    'site_input_dim': 131,
    'masf_input_dim': 5,
    'Affsrc': 'all',
}


TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 32,
    'max_epoch': 100,
    'early_stop': 20,
    'learning_rate': 5e-4,
    'weight_decay': 0,
    'step_size': 20,
    'gamma': 0.5,
    'list_task': ['Pocket'],
    'task_eval': {'Pocket': 'cls'},
    'task': 'Pocket',
    'goal': 'auroc',
}


class Model(nn.Module):
    """
    PocketAnchor for pocket detection
    """
    def __init__(self, params):
        super(Model, self).__init__()  
        """hyper part"""
        self.GNNsite_depth = int(params['GNNsite_depth'])
        self.GNNmasf_depth = int(params['GNNmasf_depth'])
        self.GNNanch_depth = int(params['GNNanch_depth'])
        self.hidden_site = int(params['hidden_site'])
        self.hidden_masf = int(params['hidden_masf'])
        self.hidden_anch = int(params['hidden_anch'])
        self.site_input_dim = int(params['site_input_dim'])
        self.masf_input_dim = int(params['masf_input_dim'])
        self.Affsrc = params['Affsrc']
        
        """model part"""
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
            nn.Linear(self.site_input_dim+self.GNNsite_depth*self.hidden_site, self.hidden_anch),
            nn.LeakyReLU(0.1),
        )
        self.AMMMasf_output = nn.Sequential(
            nn.Linear(self.masf_input_dim+self.GNNmasf_depth*self.hidden_masf, self.hidden_anch),
            nn.LeakyReLU(0.1),
        )

        # Anchor Aggregate Module parameters
        self.AAM = nn.ModuleDict()
        self.AAM["Atom"] = UpwardConv() 
        self.AAM["Masf"] = UpwardConv() 

        # Anchor MPNN
        self.AMPNN = nn.ModuleList()
        for i in range(self.GNNanch_depth):
            if i == 0:
                if self.Affsrc == "all":
                    self.AMPNN.append(AnchorConv(self.hidden_anch*2, self.hidden_anch))
                else:
                    self.AMPNN.append(AnchorConv(self.hidden_anch, self.hidden_anch))

            else:
                self.AMPNN.append(AnchorConv(self.hidden_anch, self.hidden_anch))

        if self.Affsrc == "all":
            self.AMPNN_output = nn.Sequential(
                nn.Linear(self.hidden_anch*2+self.GNNanch_depth*self.hidden_anch, self.hidden_anch),
                nn.LeakyReLU(0.1),
            )
        else:
            self.AMPNN_output = nn.Sequential(
                nn.Linear(self.hidden_anch+self.GNNanch_depth*self.hidden_anch, self.hidden_anch),
                nn.LeakyReLU(0.1),
            )

        # Classifier
        self.W_pocket = nn.Sequential(
            nn.Linear(self.hidden_anch, self.hidden_anch),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_anch, 1),
        )


    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

        self.loss = {"Pocket": nn.BCEWithLogitsLoss()}
        self.task_eval = train_params['task_eval']

    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
            loss = loss + self.loss[task](dict_pred[task], dict_label[task])
        return loss

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

    def Pocket_pred_module(self, anch_feature):
        pocket_pred = self.W_pocket(anch_feature)
        return pocket_pred

    def forward(self, siteGraphBatch, masfGraphBatch, anchGraphBatch, protGraphBatch, promGraphBatch):
        calcTime = False
        list_time = []
        device = siteGraphBatch.x.device

        # Step 1. Pocket GNN
        """
        Graph message passing on proteins
        site_feature: protein atoms
        masf_feature: points sampled from protein surface with features named MaSIF
        """
        # if calcTime: list_time.append(time.time())
        site_feature = self.AtomConv_module(siteGraphBatch)
        masf_feature = self.MasfConv_module(masfGraphBatch)
        
        # Step 2. Anchor GNN
        """
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
        
        # Step 3. Pocket prediction
        """
        predict is pocket for each anchor
        """
        if calcTime: list_time.append(time.time())
        pocket_pred = self.Pocket_pred_module(anch_feature)
        
        # Step 4. Output
        if calcTime: list_time.append(time.time())
        if calcTime: 
            list_time = np.array(list_time)
            time_elapsed = list_time[1:]-list_time[:-1]
            print(("{:.4f}\t"*(len(list_time)-1)).format(*time_elapsed))

        return {"Pocket": pocket_pred}

