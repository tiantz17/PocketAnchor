import logging
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import KFold

# load and split data
class Split(object):
    
    def split_data(self, seed, num_fold):
        if 'tabu' not in self.kwargs:
            self.kwargs['tabu'] = False
        if 'dist' not in self.kwargs:
            self.kwargs['dist'] = None
        if 'setting' not in self.kwargs:
            self.kwargs['setting'] = 'Random'
        self.split_data_fold(seed, num_fold, self.kwargs["setting"], self.kwargs["dist"], tabu=self.kwargs['tabu'])

    def split_data_fold(self, seed, num_fold, setting, thre, tabu=False):
        if setting == 'Random': # imputation
            self.split_data_imputation_random(seed, num_fold, tabu=tabu)
        elif setting == 'NewProtein':
            self.split_data_new_protein(seed, num_fold, thre, tabu=tabu)
        elif setting == 'NewCompound':
            self.split_data_new_compound(seed, num_fold, thre, tabu=tabu)
        elif setting == 'BothNew':
            self.split_data_both_new(seed, num_fold, thre, tabu=tabu)
        else:
            raise NotImplementedError
    
    def split_data_new_compound(self, seed, num_fold, thre, tabu=False):
        logging.info("Using new compound split")
        columns = 'c_group_'+str(thre)
        if columns not in self.table.columns:
            columns = 'cid'

        index_cv = np.array(self.index_all)
        list_compound = list(set(self.table.loc[self.index_all, columns].values))
        if tabu:
            list_tabu = []
            count_table = pd.value_counts(self.table['represent_pdbid'])
            for i in self.index_all:
                refid = self.table.loc[i, 'represent_pdbid']
                if count_table[refid] == 1:
                    list_tabu.append(i)
        
        np.random.seed(seed)
        np.random.shuffle(list_compound)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(list_compound):
            list_train_idx, list_valid_idx, list_test_idx = [], [], []
            test_cp = set(np.array(list_compound)[test_idx].tolist())
            train_cp = np.array(list_compound)[train_idx]
            valid_cp = set(np.random.choice(train_cp, int(len(train_cp)/num_fold), replace=False).tolist())
            for idx in index_cv:
                if self.table.loc[idx, columns] in test_cp:
                    list_test_idx.append(idx)
                elif self.table.loc[idx, columns] in valid_cp:
                    list_valid_idx.append(idx)
                else:
                    list_train_idx.append(idx)
            
            list_valid_idx = list(set(list_valid_idx))
            list_test_idx = list(set(list_test_idx))
            if tabu:
                list_valid_idx = list(set(list_valid_idx)-set(list_tabu))
                list_test_idx = list(set(list_test_idx)-set(list_tabu))
            
            self.list_fold_train.append(np.array(list_train_idx))
            self.list_fold_valid.append(np.array(list_valid_idx))
            self.list_fold_test.append(np.array(list_test_idx))
        
    def split_data_new_protein(self, seed, num_fold, thre, tabu=False):
        logging.info("Using new protein split")
        columns = 'p_group_'+str(thre)
        if columns not in self.table.columns:
            columns = 'pid'
            
        index_cv = np.array(self.index_all)
        list_protein = list(set(self.table.loc[self.index_all, columns].values))
        if tabu:
            list_tabu = []
            count_table = pd.value_counts(self.table['represent_pdbid'])
            for i in self.index_all:
                refid = self.table.loc[i, 'represent_pdbid']
                if count_table[refid] == 1:
                    list_tabu.append(i)
            
        np.random.seed(seed)
        np.random.shuffle(list_protein)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(list_protein):
            list_train_idx, list_valid_idx, list_test_idx = [], [], []
            test_pid = set(np.array(list_protein)[test_idx].tolist())
            train_pid = np.array(list_protein)[train_idx]
            valid_pid = set(np.random.choice(train_pid, int(len(train_pid)/num_fold), replace=False).tolist())
            for idx in index_cv:
                if self.table.loc[idx, columns] in test_pid:
                    list_test_idx.append(idx)
                elif self.table.loc[idx, columns] in valid_pid:
                    list_valid_idx.append(idx)
                else:
                    list_train_idx.append(idx)
            
            list_valid_idx = list(set(list_valid_idx))
            list_test_idx = list(set(list_test_idx))
            if tabu:
                list_valid_idx = list(set(list_valid_idx)-set(list_tabu))
                list_test_idx = list(set(list_test_idx)-set(list_tabu))
            
            self.list_fold_train.append(np.array(list_train_idx))
            self.list_fold_valid.append(np.array(list_valid_idx))
            self.list_fold_test.append(np.array(list_test_idx))
    
    def split_data_both_new(self, seed, num_fold, thre, tabu=False):
        assert np.sqrt(num_fold) == int(np.sqrt(num_fold))
        
        c_columns = 'c_group_'+str(thre)
        p_columns = 'p_group_'+str(thre)
        if c_columns not in self.table.columns:
            c_columns = 'cid'
            p_columns = 'pid'
        logging.info("Using both-new split")
                
        index_cv = np.array(self.index_all)
        list_compound = list(set(self.table.loc[self.index_all, c_columns].values))
        list_protein = list(set(self.table.loc[self.index_all, p_columns].values))
        if tabu:
            list_tabu = []
            count_table = pd.value_counts(self.table['represent_pdbid'])
            for i in self.index_all:
                refid = self.table.loc[i, 'represent_pdbid']
                if count_table[refid] == 1:
                    list_tabu.append(i)
                
        np.random.seed(seed)
        np.random.shuffle(list_compound)
        np.random.seed(seed)
        np.random.shuffle(list_protein)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kfp = KFold(n_splits=int(np.sqrt(num_fold)), shuffle=True)
        kfc = KFold(n_splits=int(np.sqrt(num_fold)), shuffle=True)
        cpd_split = []
        for train_idx_c, test_idx_c in kfc.split(list_compound):
            cpd_split.append([train_idx_c, test_idx_c])

        for train_idx_p, test_idx_p in kfp.split(list_protein):
            for fold in range(len(cpd_split)):
                train_idx_c, test_idx_c = cpd_split[fold]

                list_train_idx, list_valid_idx, list_test_idx = [], [], []
                test_pid = set(np.array(list_protein)[test_idx_p].tolist())
                train_pid = np.array(list_protein)[train_idx_p]
                valid_pid = set(np.random.choice(train_pid, int(len(train_pid)/np.sqrt(num_fold)), replace=False).tolist())

                test_cid = set(np.array(list_compound)[test_idx_c].tolist())
                train_cid = np.array(list_compound)[train_idx_c]
                valid_cid = set(np.random.choice(train_cid, int(len(train_cid)/np.sqrt(num_fold)), replace=False).tolist())

                for idx in index_cv:
                    if self.table.loc[idx, p_columns] in test_pid and self.table.loc[idx, c_columns] in test_cid:
                        list_test_idx.append(idx)
                    elif self.table.loc[idx, p_columns] in valid_pid and self.table.loc[idx, c_columns] in valid_cid:
                        list_valid_idx.append(idx)
                    elif self.table.loc[idx, p_columns] in train_pid and self.table.loc[idx, c_columns] in train_cid:
                        list_train_idx.append(idx)
                        
                list_valid_idx = list(set(list_valid_idx))
                list_test_idx = list(set(list_test_idx))
                if tabu:
                    list_valid_idx = list(set(list_valid_idx)-set(list_tabu))
                    list_test_idx = list(set(list_test_idx)-set(list_tabu))
            
                self.list_fold_train.append(np.array(list_train_idx))
                self.list_fold_valid.append(np.array(list_valid_idx))
                self.list_fold_test.append(np.array(list_test_idx))

    def split_data_imputation_random(self, seed, num_fold, tabu=False):
        logging.info("Using new interaction random split")
        np.random.seed(seed)
        
        index_cv = np.array(self.index_all)
        if tabu:
            list_tabu = []
            count_table = pd.value_counts(self.table['represent_pdbid'])
            for i in self.index_all:
                refid = self.table.loc[i, 'represent_pdbid']
                if count_table[refid] == 1:
                    list_tabu.append(i)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(index_cv):
            list_valid_idx = np.random.choice(train_idx, int(len(train_idx)/num_fold), replace=False)
            list_train_idx = list(set(train_idx.tolist())-set(list_valid_idx.tolist()))
            list_test_idx = test_idx
            
            list_valid_idx = list(set(list_valid_idx.tolist()))
            list_test_idx = list(set(list_test_idx.tolist()))
            if tabu:
                list_valid_idx = list(set(list_valid_idx)-set(list_tabu))
                list_test_idx = list(set(list_test_idx)-set(list_tabu))
                
            self.list_fold_train.append(index_cv[list_train_idx])
            self.list_fold_valid.append(index_cv[list_valid_idx])
            self.list_fold_test.append(index_cv[list_test_idx])
    

# evaluation functions
class Eval(object):
    """
    Class for evaluation methods
    """
    def score(self, scoretype):
        if scoretype == "reg":
            return self.score_reg
        elif scoretype == "cls":
            return self.score_cls

    def score_reg(self, pred, label):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)
        try:
            r2 = r2_score(label, pred)
        except:
            r2 = np.nan
        try:
            mse = mean_squared_error(label, pred)
        except:
            mse = np.nan
        try:
            pcc = pearsonr(label, pred)[0]
        except:
            pcc = np.nan
        try:
            scc = spearmanr(label, pred)[0]
        except:
            scc = np.nan
        return {"r2":r2, "mse":mse, "pcc":pcc, "scc":scc}

    def score_cls(self, pred, label):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)
        try:
            auroc = roc_auc_score(label, pred)
        except:
            auroc = np.nan
        try:
            aupr = average_precision_score(label, pred)
        except:
            aupr = np.nan
        return {"auroc":auroc, "aupr":aupr}
