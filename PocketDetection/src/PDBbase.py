import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class PDBbase(Dataset):
    """
    Class for PDB based datasets

    If data loading changed, please modify the following registration:
        - register_data_table
        - register_atom_feature
        - register_masif_feature
        - register_anchor
    and get:
        - get_label
        - get_atom_features
        - get_masif_features
        - get_anchor

    """
    def __getitem__(self, idx):
        """
        load data for a single instance
        """
        # get pdbid and label
        pdbid = self.table.loc[self.index[idx], 'pdbid']
        pocket_label = self.get_label(pdbid)
        # get graph of atoms
        atomGraph = self.get_atom_graph(pdbid)
        # get graph of masifs
        masfGraph = self.get_masf_graph(pdbid)
        # get graph of anchors, including graph of atom->anchor and masif->anchor
        aa, at, am = self.get_anchor(pdbid)
        anchGraph, protGraph = self.get_prot_graph(at, aa)
        _, promGraph = self.get_prot_graph(am, aa)       

        return atomGraph, masfGraph, anchGraph, protGraph, promGraph, pocket_label, pdbid
                
    def __len__(self):
        return len(self.index)
    
    def reset_index(self, index):
        self.index = index

    def register_data(self, list_pdbid, anchor_dict):
        """
        register data from a single file or from individual files
        """
        # data and labels
        self.register_label(list_pdbid, anchor_dict)
        # features of atom level
        self.register_atom_feature()
        # features of protein surface level
        self.register_masif_feature()
        # features of anchor level
        self.register_anchor()

    def register_label(self, list_pdbid, dict_label):
        self.dict_label = dict_label
        self.list_pdbid = list_pdbid
        self.index_all = []
        for i in self.table.index:
            if self.table.loc[i, 'pdbid'] in self.list_pdbid:
                self.index_all.append(i)
        self.index = self.index_all

    def register_atom_feature(self):
        # dict containing [feature, coord, neighbor]
        self.dict_atom_feature = pickle.load(open(self.path + "atom_feature_coord_nei_dict_thre" + self.kwargs['thre'], "rb")) 

    def register_anchor(self):
        # anchor-atom
        self.dict_at_dis_mat = pickle.load(open(self.path + "at_dict_thre" + self.kwargs['thre'], "rb"))
        # anchor-anchor
        self.dict_aa_dis_mat = pickle.load(open(self.path + "aa_dict_thre" + self.kwargs['thre'], "rb"))
        # anchor-masif
        self.dict_am_dis_mat = pickle.load(open(self.path + "am_dict_thre" + self.kwargs['thre'], "rb"))

    def register_masif_feature(self):
        self.dict_masif = pickle.load(open(self.path + "masif_feature_coord_nei_dict", "rb"))

    def get_label(self, pdbid):
        if pdbid in self.dict_label:
            return torch.FloatTensor(self.dict_label[pdbid]).reshape(-1, 1)
        else:
            return None

    def get_atom_features(self, pdbid):
        return self.dict_atom_feature[pdbid]

    def get_masif_features(self, pdbid):
        masf_feature, masif_coords, masif_neighbor = self.dict_masif[pdbid]

        return masf_feature, masif_neighbor, masif_coords

    def get_anchor(self, pdbid):
        aa_distance_mat = self.dict_aa_dis_mat[pdbid]
        at_distance_mat = self.dict_at_dis_mat[pdbid]
        am_distance_map = self.dict_am_dis_mat[pdbid]

        return aa_distance_mat, at_distance_mat, am_distance_map

    def get_atom_graph(self, pdbid):
        atom_feature, atom_pos, nei_list = self.get_atom_features(pdbid)
        atom_feature = self.convert_feature(atom_feature)
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
        masf_feature, masif_neighbor, masif_coords = self.get_masif_features(pdbid)
        if len(masif_neighbor) == 0:
            masif_neighbor = np.zeros((2, 0))
        edge_index = np.concatenate([masif_neighbor, masif_neighbor[:, [1, 0]]], axis=0)
        edge_index = torch.LongTensor(edge_index).T
        masf_feature = torch.FloatTensor(masf_feature)
        masif_coords = torch.FloatTensor(masif_coords)

        # BUG
        masf_feature[masf_feature < -100] = -15

        masfGraph = Data(x=masf_feature, edge_index=edge_index, pos=masif_coords)
        if edge_index.shape[1] > 0:
            assert edge_index.max() < masfGraph.num_nodes

        return masfGraph

    def get_prot_graph(self, anchor_atom_mat, anchor_anchor_mat):
        num_anchor, num_site = anchor_atom_mat.shape
        edge_index = anchor_atom_mat.coalesce().indices()[[1, 0]]
        edge_index[1] += num_site
        edge_attr = anchor_atom_mat.coalesce().values()
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

        edge_index = anchor_anchor_mat.coalesce().indices()
        edge_attr = anchor_anchor_mat.coalesce().values()
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


