import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from common.src.datasets.base import Eval
from PocketDetection.src.PDBbase import PDBbase


DATASET_PARAMS = {
    "dist": 0.3,
    "thre": "8",
}


def batch_data_process_PocketAnchor(data):
    # re-organize
    data = list(zip(*data))
    atomGraph, masfGraph, anchGraph, protGraph, promGraph, pocket_label, pdbid = data 

    # from list to batch
    atomGraphBatch = Batch().from_data_list(atomGraph)
    masfGraphBatch = Batch().from_data_list(masfGraph)
    anchGraphBatch = Batch().from_data_list(anchGraph)
    protGraphBatch = Batch().from_data_list(protGraph)
    promGraphBatch = Batch().from_data_list(promGraph)
 
    # distance matrices
    pocket_label = torch.cat(pocket_label).view((-1,1))
    pocket_label = torch.FloatTensor(pocket_label)
    
    return (atomGraphBatch, masfGraphBatch, anchGraphBatch, protGraphBatch, promGraphBatch), {"Pocket": pocket_label, "PDBID": pdbid}
    

class DataSet(PDBbase, Eval):
    """
    Dataset for HOLO4k
    """
    def __init__(self, path, kwargs):
        self.path = path + "anchor_pocket_holo/"
        self.kwargs = kwargs
        # load data
        self.table = pd.read_csv(self.path + "holo4k_table_pocket.tsv", sep='\t')
        dict_label = pickle.load(open(self.path + "anchor_label_n4_dict_" + self.kwargs['thre'], "rb"))

        list_pdbid = list(self.table['pdbid'])
        self.register_data(list_pdbid, dict_label)
        