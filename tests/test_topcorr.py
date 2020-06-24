
import unittest
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.rinterface as rinterface
from numpy.testing import assert_array_almost_equal, assert_equal
import topcorr
import networkx as nx
import planarity

class TestTopCorr(unittest.TestCase):
    def _construct_tmfg_with_r(self, corr):
        """
        Constructs a TMFG using the implementation provided
        by the authors
        """
        rpy2.robjects.numpy2ri.activate()
        nt = importr('NetworkToolbox')
        tmfg_corr = nt.TMFG(corr, "pairwise")
        return np.array(tmfg_corr[0])

    def _construct_dependency_network_with_r(self, X):
        """
        Constructs a dependency network from the NetworkToolbox R
        package
        """
        rpy2.robjects.numpy2ri.activate()
        nt = importr('NetworkToolbox')
        depend_corr = nt.depend(X, False, "none")
        return np.array(depend_corr)

    def test_tmfg(self):
        """
        Compares out TMFG algorithm to the one in the Network Toolbox
        """
        p = 50
        mean = np.zeros(p)
        
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        #C = np.abs(corr)
        G = topcorr.tmfg(corr, absolute=True)
        tmfg_r = self._construct_tmfg_with_r(corr)
        tmfg_r[np.abs(tmfg_r) > 0] = 1
        tmfg_me = nx.to_numpy_array(G, weight=None)
        np.fill_diagonal(tmfg_me, 1)
        assert_array_almost_equal(tmfg_r, tmfg_me)

    def test_pmfg(self):
        """
        Tests the PMFG - the way we're going to do this is by generating a
        correlation matrix that is planar and see if the algorithm can pick it up.
        In this case we choose a goldner-harary graph https://en.wikipedia.org/wiki/Goldner%E2%80%93Harary_graph
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
        #corr[corr < 0.1] = 0
        corr_G = nx.from_numpy_array(corr)
        pmfg_G = topcorr.pmfg(corr)
        corr_pmfg = nx.to_numpy_array(pmfg_G, weight=None)
        assert_array_almost_equal(corr_pmfg, corr_true_binary)

    def test_mst(self):
        """
        Tests the MST by comparing it to the networkx implementation
        """
        p = 50
        mean = np.zeros(p)
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        nodes = list(np.arange(p))
        # For the networkx MST we have to convert the correlation graph
        # into a distance one
        D = np.sqrt(2 - 2*corr)
        G = nx.from_numpy_array(D)
        mst_G = nx.minimum_spanning_tree(G)

        topcorr_mst_G = topcorr.mst(corr)

        mst_nx_M = nx.to_numpy_array(mst_G, nodelist=nodes, weight=None)
        mst_topcorr_M = nx.to_numpy_array(topcorr_mst_G, nodelist=nodes, weight=None)

        assert_array_almost_equal(mst_nx_M, mst_topcorr_M)

    def test_threshold(self):
        """
        Tests the thresholding of the correlation matrix - here we add a 
        bit of noise to a generated sparse matrix and see if we can recover 
        the non zeros
        """
        p = 50
        mean = np.zeros(p)
        M = make_sparse_spd_matrix(p, alpha=0.95, norm_diag = True, smallest_coef=0.7)
        t = np.abs(M).min()
        noise = 0.5*t * np.random.rand(p, p)
        M_noise = noise + M
        threshold = topcorr.threshold(M_noise, t, binary=True)
        M[np.abs(M) > 0] = 1
        assert_array_almost_equal(M, threshold)

    def test_dependency(self):
        """
        Tests the dependency network by comparing it to the NetworkToolbox method
        """
        p = 50
        mean = np.zeros(p)
        M = make_sparse_spd_matrix(p, alpha=0.95, norm_diag = True, smallest_coef=0.7)

        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)

        D_topcorr = topcorr.dependency_network(corr)
        D_networktoolbox = self._construct_dependency_network_with_r(X)

        assert_array_almost_equal(D_networktoolbox, D_topcorr)

    def test_knn(self):
        """
        Tests the kNN network by ensuring the resulting network meets the constraints
        """
        p = 5
        k = 2
        mean = np.zeros(p)
        M = make_sparse_spd_matrix(p, alpha=0.95, norm_diag = True, smallest_coef=0.7)

        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)

        G = topcorr.knn(corr, k)
        corr_knn = nx.to_numpy_array(G)

        for i in range(p):
            assert(np.count_nonzero(corr_knn[:, i]) >= k)

        assert(np.count_nonzero(corr_knn) < 2*k*p)

    def test_partial_correlation(self):
        """
        Tests the partial correlation network by ensuring the resulting partial correlation
        matrix correctly recovers the nonzeros from a sparse underlying precision matrix
        """
        p = 10
        mean = np.zeros(p)
        K = make_sparse_spd_matrix(p, alpha=0.9, norm_diag = True, smallest_coef=0.7)
        C = np.linalg.inv(K)
        ind = np.nonzero(K)
        t = 0.8*np.abs(K[ind]).min()

        partial_correlation = topcorr.partial_correlation(C)

        threshold = topcorr.threshold(partial_correlation, t, binary=True)
        K[np.abs(K) > 0] = 1
        assert_array_almost_equal(K, threshold)

    def test_affintiy(self):
        """
        Tests the affinity matrix method by ensuring that it runs. Currently a bit
        stumped for a good test.
        """
        p = 10
        mean = np.zeros(p)
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)

        A = topcorr.affinity(corr)