
import unittest
from sklearn.datasets import make_spd_matrix
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.rinterface as rinterface
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
import topcorr
import networkx as nx

class TestToppCorr(unittest.TestCase):
    def _construct_tmfg_with_r(self, corr):
        """
        Constructs a TMFG using the implementation provided
        by the authors
        """
        rpy2.robjects.numpy2ri.activate()
        nt = importr('NetworkToolbox')
        tmfg_corr = nt.TMFG(corr, "pairwise")
        return np.array(tmfg_corr[0])

    def test_tmfg(self):
        """
        Compares out TMFG algorithm to the one in the Network Toolbox
        """
        p = 50
        mean = np.zeros(p)
        
        M = make_spd_matrix(p,random_state=1)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        C = np.abs(corr)
        G = topcorr.tmfg(C)
        tmfg_r = self._construct_tmfg_with_r(corr)
        tmfg_r[np.abs(tmfg_r) > 0] = 1
        tmfg_me = nx.to_numpy_array(G, weight=None)
        np.fill_diagonal(tmfg_me, 1)
        assert_array_almost_equal(tmfg_r, tmfg_me)