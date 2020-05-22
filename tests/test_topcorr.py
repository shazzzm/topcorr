
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
import planarity

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
        
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        C = np.abs(corr)
        G = topcorr.tmfg(C)
        tmfg_r = self._construct_tmfg_with_r(C)
        tmfg_r[np.abs(tmfg_r) > 0] = 1
        tmfg_me = nx.to_numpy_array(G, weight=None)
        np.fill_diagonal(tmfg_me, 1)
        assert_array_almost_equal(tmfg_r, tmfg_me)

    def test_pmfg(self):
        """
        Tests the PMFG - the way we're going to do this is by generating a
        correlation matrix that is planar and see if the algorithm can pick it up.
        In this case we choose a goldner-harary graph
        """
        p = 11
        mean = np.zeros(p)

        e= [(1,2 ),( 1,3 ),( 1,4 ),( 1,5 ),( 1,7 ),( 1,8 ),( 1,10 ),
            ( 1,11 ),( 2,3 ),( 2,4 ),( 2,6 ),( 2,7 ),( 2,9 ),( 2,10 ),
            ( 2,11 ),( 3,4 ),( 4,5 ),( 4,6 ),( 4,7 ),( 5,7 ),( 6,7 ),
            ( 7,8 ),( 7,9 ),( 7,10 ),( 8,10 ),( 9,10 ),( 10,11)]
        true_G = nx.Graph(e)
        corr_true_binary = nx.to_numpy_array(true_G, weight=None)
        corr_true = corr_true_binary.copy() * 0.5
        np.fill_diagonal(corr_true, 1)

        X = np.random.multivariate_normal(mean, corr_true, 2000)
        corr = np.corrcoef(X.T)
        corr[corr < 0.1] = 0
        corr_G = nx.from_numpy_array(corr)
        pmfg_G = topcorr.pmfg(corr)
        corr_pmfg = nx.to_numpy_array(pmfg_G, weight=None)
        assert_array_almost_equal(corr_pmfg, corr_true_binary)

