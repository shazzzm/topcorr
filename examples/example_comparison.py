"""
This script gives an example of how the various methods in 
TopCorr differ in their filtration of the same correlation
matrix
"""
import numpy as np
import networkx as nx
from sklearn.datasets import make_spd_matrix
import topcorr

p = 50
n = 200
M = make_spd_matrix(p)
X = np.random.multivariate_normal(np.zeros(p), M, 200)
corr = np.corrcoef(X.T)
nodes = list(np.arange(p))

topcorr_mst = topcorr.mst(corr)
topcorr_pmfg = topcorr.pmfg(corr)
topcorr_tmfg = topcorr.tmfg(corr)
topcorr_threshold = nx.from_numpy_array(topcorr.threshold(corr, 0.2))

print("MST edges: %s" % len(topcorr_mst.edges()))
print("PMFG edges: %s" % len(topcorr_pmfg.edges()))
print("TMFG edges: %s" % len(topcorr_tmfg.edges()))
print("Threshold edges: %s" % len(topcorr_threshold.edges()))