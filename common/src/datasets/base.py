import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score


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
