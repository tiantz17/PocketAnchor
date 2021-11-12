import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from common.src.datasets.base import Eval
from common.src.datasets.utils import ATOM_FDIM, BOND_FDIM


DATASET_PARAMS = {
    "dist": 0.3,
    "thre": "4",
}


def batch_data_process_PocketAnchor(data):
    # re-organize
    data = list(zip(*data))
    vertGraph, grouGraph, compGraph, atomGraph, masfGraph, anchGraph, protGraph, promGraph, tc_mat, pdbid = data 

    # from list to batch
    vertGraphBatch = Batch().from_data_list(vertGraph)
    grouGraphBatch = Batch().from_data_list(grouGraph)
    compGraphBatch = Batch().from_data_list(compGraph)
    atomGraphBatch = Batch().from_data_list(atomGraph)
    masfGraphBatch = Batch().from_data_list(masfGraph)
    anchGraphBatch = Batch().from_data_list(anchGraph)
    protGraphBatch = Batch().from_data_list(protGraph)
    promGraphBatch = Batch().from_data_list(promGraph)

    # # distance matrices
    list_tc = torch.cat([torch.cat([tc.to_dense().view(-1) for tc in tc_list]).view(-1, 1) for tc_list in list(zip(*tc_mat))], dim=1)
    list_tc = list_tc * (list_tc>0) + torch.exp(list_tc) * (list_tc<0)

 
    return (vertGraphBatch, grouGraphBatch, compGraphBatch, atomGraphBatch, \
        masfGraphBatch, anchGraphBatch, protGraphBatch, promGraphBatch), \
            {
                "Interaction":list_tc, \
                "PDBID": pdbid}
    

class DataSet(Dataset, Eval):
    """
    Dataset for PDBbind 2016/2020, CASF 2016 [already preprocessed]
    """
    def __init__(self, path, kwargs):
        self.path = path + "anchor_casf/"
        self.kwargs = kwargs
        # load data
        self.table = pd.read_csv(self.path + "casf2016_table_new_protein.tsv", sep='\t')

        list_datatype = ["test", "expand"]
        
        if self.kwargs["datatype"] == "test":
            index = self.table["type"] == "test"
        elif self.kwargs["datatype"] == "expand":
            index = self.table["type"] == "test"
            index |= self.table["type"] == "np"
        elif self.kwargs["datatype"] not in list_datatype:
            raise NotImplementedError

        self.table = self.table[index]
        self.table.reset_index(inplace=True, drop=True)

        list_pdbid = self.table['pdbid'].tolist()
        self.register_data(list_pdbid)
    
    def __getitem__(self, idx):
        """
        load data for a single instance
        """
        # get pdbid and label
        pdbid = self.table.loc[self.index[idx], 'pdbid']

        # get graph of compound atoms
        vertGraph = self.get_vert_graph(pdbid)

        # get graph of protein atoms
        atomGraph = self.get_atom_graph(pdbid)
        # get graph of protein masifs
        masfGraph = self.get_masf_graph(pdbid)

        # get graph of compound groups, including graph of atom->group
        gt,tc = self.get_group(pdbid)
        grouGraph, compGraph = self.get_comp_graph(gt, vertGraph.num_nodes)
        # get graph of protein anchors, including graph of atom->anchor and masif->anchor
        aa, at, am = self.get_anchor(pdbid)
        anchGraph, protGraph = self.get_prot_graph(at, aa)
        _, promGraph = self.get_prot_graph(am, aa)

        return vertGraph, grouGraph, compGraph, atomGraph, masfGraph, anchGraph, protGraph, promGraph, tc, pdbid
                
    def __len__(self):
        return len(self.index)
    
    def reset_index(self, index):
        self.index = index

    def register_data(self, list_pdbid):
        """
        register data from a single file or from individual files
        """
        # data and labels
        self.register_label(list_pdbid)
        # init features
        self.register_init_feature()
        # features of vert level
        self.register_vert_feature()
        # features of atom level
        self.register_atom_feature()
        # features of protein surface level
        self.register_masif_feature()
        # features of anchor level
        self.register_anchor()
        # features of group level
        self.resigter_group()

    def register_label(self, list_pdbid):
        self.list_pdbid = list_pdbid
        self.index_all = []
        for i in self.table.index:
            if self.table.loc[i, 'pdbid'] in self.list_pdbid:
                self.index_all.append(i)
        self.index = self.index_all

    def register_init_feature(self):
        #load intial atom and bond features (i.e., embeddings)
        with open(self.path+'casf2016_atom_dict', 'rb') as f:
            atom_dict = pickle.load(f)
        with open(self.path+'casf2016_bond_dict', 'rb') as f:
            bond_dict = pickle.load(f)
        
        init_atom_features = np.zeros((len(atom_dict), ATOM_FDIM))
        init_bond_features = np.zeros((len(bond_dict), BOND_FDIM))
        
        for key,value in atom_dict.items():
            init_atom_features[value] = np.array(list(map(int, key)))
        
        for key,value in bond_dict.items():
            init_bond_features[value] = np.array(list(map(int, key)))

        self.init_atom_features = torch.FloatTensor(init_atom_features)
        self.init_bond_features = torch.FloatTensor(init_bond_features)

    def register_vert_feature(self):
        # dict containing [vertex, edge, atom_adj, bond_adj, nbs]
        self.dict_vert_feature = pickle.load(open(self.path + "casf2016_cf_dict", "rb")) 

    def register_atom_feature(self):
        # dict containing [feature, coord, neighbor]
        self.dict_atom_feature = pickle.load(open(self.path + "casf2016_atom_feature_coord_nei_dict_thre" + self.kwargs['thre'], "rb"))

    def resigter_group(self):
        # group-atom
        self.dict_gt_mat = pickle.load(open(self.path + "casf2016_frag_dict", "rb"))
        # atom-atom
        self.dict_tc_int_mat = pickle.load(open(self.path + "casf2016_tc_dict", "rb"))

    def register_anchor(self):
        # anchor-atom
        self.dict_at_dis_mat = pickle.load(open(self.path + "casf2016_at_dict_thre" + self.kwargs['thre'], "rb"))
        # anchor-anchor
        self.dict_aa_dis_mat = pickle.load(open(self.path + "casf2016_aa_dict_thre" + self.kwargs['thre'], "rb"))
        # anchor-masif
        self.dict_am_dis_mat = pickle.load(open(self.path + "casf2016_am_dict_thre" + self.kwargs['thre'], "rb"))

    def register_masif_feature(self):
        self.dict_masif_feature = pickle.load(open(self.path + "casf2016_masif_feature_coord_nei_dict", "rb"))

    def get_label(self, pdbid):
        if pdbid in self.dict_label:
            return torch.FloatTensor(self.dict_label[pdbid]).reshape(-1, 1)
        else:
            return None

    def get_vert_features(self, pdbid):
        return self.dict_vert_feature[pdbid]

    def get_atom_features(self, pdbid):
        return self.dict_atom_feature[pdbid]

    def get_masif_features(self, pdbid):
        masif_feature, _, masif_neighbor = self.dict_masif_feature[pdbid]
        return masif_feature, masif_neighbor

    def get_group(self, pdbid):
        gt_mat = self.dict_gt_mat[pdbid]
        tc_mat = self.dict_tc_int_mat[pdbid]

        return gt_mat, tc_mat

    def get_anchor(self, pdbid):
        aa_distance_mat = self.dict_aa_dis_mat[pdbid]
        at_distance_mat = self.dict_at_dis_mat[pdbid]
        am_distance_map = self.dict_am_dis_mat[pdbid]

        return aa_distance_mat, at_distance_mat, am_distance_map

    def get_vert_graph(self, pdbid):
        vertex, edge, atom_adj, bond_adj, nbs = self.get_vert_features(pdbid)
        node_initial = self.init_atom_features[vertex]
        edge_initial = self.init_bond_features[edge]
        edge_index = []
        edge_attr = []
        for i, j, k in zip(np.where(nbs==1)[0], atom_adj[nbs==1], bond_adj[nbs==1]):
            edge_index.append([i, j]) 
            edge_attr.append(edge_initial[k:k+1])
            edge_index.append([j, i]) 
            edge_attr.append(edge_initial[k:k+1])
        if len(edge_index) == 0:
            edge_index = torch.empty(0, 2).long()
            edge_attr = torch.empty(0, BOND_FDIM)
        else:
            edge_attr = torch.cat(edge_attr)
        edge_index = torch.LongTensor(edge_index).T
        vertGraph = Data(x=node_initial, edge_index=edge_index, edge_attr=edge_attr)

        return vertGraph

    def get_atom_graph(self, pdbid):
        atom_feature, atom_pos, nei_list = self.get_atom_features(pdbid)
        # atom_feature = self.convert_feature(atom_feature)
        # assert type(nei_list) == list
        edge_index = []
        for i, neis in enumerate(nei_list):
            for j in neis:
                edge_index.append([i, j])
                edge_index.append([j, i])
        if len(edge_index) == 0:
            edge_index = torch.empty(0, 2).long()
        edge_index = torch.LongTensor(edge_index).T
        atom_feature = torch.FloatTensor(atom_feature)
        atom_pos = torch.FloatTensor(atom_pos)

        atomGraph = Data(x=atom_feature, edge_index=edge_index, pos=atom_pos)

        return atomGraph

    def get_masf_graph(self, pdbid):
        masf_feature, masif_neighbor = self.get_masif_features(pdbid)
        if len(masif_neighbor) == 0:
            masif_neighbor = np.zeros((0, 2))
        edge_index = np.concatenate([masif_neighbor, masif_neighbor[:, [1, 0]]], axis=0)
        edge_index = torch.LongTensor(edge_index).T
        masf_feature = torch.FloatTensor(masf_feature)

        # BUG
        masf_feature[masf_feature < -100] = -15

        masfGraph = Data(x=masf_feature, edge_index=edge_index)
        if len(masif_neighbor) > 0:
            assert edge_index.max() < masfGraph.num_nodes

        return masfGraph

    def get_comp_graph(self, group_atom_mat, num_atom):
        num_group = len(group_atom_mat)
        edge_index = []
        for i, group in enumerate(group_atom_mat):
            for j in group:
                edge_index.append([j, i+num_atom])
        edge_index = torch.LongTensor(edge_index).T
        kwargs = {
            'num_anchor': num_group,
            'num_site': num_atom
            }
        mask = torch.cat([torch.ones(num_atom), torch.zeros(num_group)]).bool()
        compGraph = Data(edge_index=edge_index, kwargs=kwargs, mask=mask)
        compGraph.num_nodes = num_atom + num_group

        grouGraph = Data(x=torch.empty(num_group, 0))

        return grouGraph, compGraph

    def get_prot_graph(self, anchor_atom_mat, anchor_anchor_mat):
        num_anchor, num_site = anchor_atom_mat.shape
        # edge_index = anchor_atom_mat.coalesce().indices()[[1, 0]]
        edge_index = np.vstack(np.where(anchor_atom_mat < 6))[[1, 0]]
        edge_index[1] += num_site
        # edge_attr = anchor_atom_mat.coalesce().values()
        edge_attr = anchor_atom_mat[np.where(anchor_atom_mat < 6)]
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr).reshape(-1, 1)
        edge_attr = torch.exp(-edge_attr/2)

        kwargs = {
            'num_anchor': num_anchor,
            'num_site': num_site,
            }
        mask = torch.cat([torch.ones(num_site), torch.zeros(num_anchor)]).bool()
        protGraph = Data(edge_index=edge_index, edge_attr=edge_attr, kwargs=kwargs, mask=mask)
        protGraph.num_nodes = num_site + num_anchor
        
        if edge_index.shape[1] > 0:
            assert edge_index.max() < protGraph.num_nodes

        # edge_index = anchor_anchor_mat.coalesce().indices()
        # edge_attr = anchor_anchor_mat.coalesce().values()
        edge_index = torch.LongTensor(np.vstack(np.where(anchor_anchor_mat < 6)))
        edge_attr = torch.FloatTensor(anchor_anchor_mat[np.where(anchor_anchor_mat < 6)])
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr])
        
        index = (edge_attr > 0) & (edge_attr < 6)
        edge_index = edge_index[:, index]
        edge_attr = edge_attr[index]
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr).reshape(-1, 1)
        edge_attr = torch.exp(-edge_attr/2)

        anchGraph = Data(x=torch.empty(num_anchor, 0), edge_index=edge_index, edge_attr=edge_attr)
        if edge_index.shape[1] > 0:
            assert edge_index.max() < anchGraph.num_nodes

        return anchGraph, protGraph

    def convert_feature(self, feature):
        feature = torch.Tensor(feature)
        num_atom = feature.shape[0]
        value_features = feature[:, 4:]
        index = torch.arange(num_atom)
        atom_elem = torch.zeros(num_atom, 6)
        atom_elem[index, feature[:, 0].long()] = 1
        aa_type = torch.zeros(num_atom, 21)
        aa_type[index, feature[:, 1].long()] = 1
        atom_in_aa = torch.zeros(num_atom, 93)
        atom_in_aa[index, feature[:, 2].long()] = 1
        atom_in_aa[feature[:, 2] == -1] = 0
        ss_type = torch.zeros(num_atom, 4)
        ss_type[index, feature[:, 3].long()] = 1
        return torch.cat([atom_elem, aa_type, atom_in_aa, ss_type, value_features], dim=1)
