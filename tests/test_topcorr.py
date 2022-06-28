
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
        tmfg_corr = nt.TMFG(corr)
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
        G = topcorr.tmfg(corr, absolute=True, threshold_mean=True)
        tmfg_r = self._construct_tmfg_with_r(corr)
        tmfg_r[np.abs(tmfg_r) > 0] = 1
        tmfg_me = nx.to_numpy_array(G, weight=None)
        np.fill_diagonal(tmfg_me, 1)
        assert_array_almost_equal(tmfg_r, tmfg_me)

        G = topcorr.tmfg(corr, threshold_mean=False)
        assert(nx.check_planarity(G)[0])

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

    def test_prim(self):
        """
        Tests the implementation of Prim's algorithm by comparing it to the networkx
        implementation
        """
        p = 10
        mean = np.zeros(p)
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        nodes = list(np.arange(p))
        # For the networkx MST we have to convert the correlation graph
        # into a distance one
        D = np.sqrt(2 - 2*corr)
        G = nx.from_numpy_array(D)
        mst_G = nx.minimum_spanning_tree(G, algorithm="prim")

        topcorr_mst_G = topcorr.mst(corr, algorithm="prim")

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

    def test_al_mst(self):
        """
        Tests the average linkage MST by ensuring that it runs
        """
        p = 200
        mean = np.zeros(p)
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        nodes = list(np.arange(p))
        # For the networkx MST we have to convert the correlation graph
        # into a distance one

        topcorr_mst_G = topcorr.almst(corr)

        mst_topcorr_M = nx.to_numpy_array(topcorr_mst_G, nodelist=nodes, weight=None)

    def test_forest(self):
        """
        Tests the forest construction by firstly ensuring the MSTs are identical when 
        the correlation matrix only has unique edges, and secondly when the correlation
        matrix is degenerate
        """
        p = 10
        mean = np.zeros(p)
        M = make_spd_matrix(p)
        X = np.random.multivariate_normal(mean, M, 200)
        corr = np.corrcoef(X.T)
        nodes = list(np.arange(p))
        mst = topcorr.mst(corr)
        forest = topcorr.mst_forest(corr)

        M_mst = nx.to_numpy_array(mst, nodes)
        M_forest = nx.to_numpy_array(forest, nodes)

        assert_array_almost_equal(M_mst, M_forest)

        example_mat = np.array([[0, 0.1, 0.3, 0.2, 0.1], [0.1, 0, 0.3, 0.4, 1.7], [0.3, 0.3, 0, 0.6, 0.5], [0.2, 0.4, 0.6, 0, 0.2], [0.1, 1.7, 0.5, 0.2, 0]])
        example_corr = 1-np.power(example_mat,2)/2 
        forest = topcorr.mst_forest(example_corr)
        mst = topcorr.mst(example_corr)
        forest_edges = len(forest.edges)
        mst_edges = len(mst.edges)

        assert(forest_edges > mst_edges)
        assert(nx.is_connected(forest))

    def test_dcca(self):
        """
        Tests the DCCA method using the data provided by https://gist.github.com/jaimeide/a9cba18192ee904307298bd110c28b14
        """
        x1 = [-1.042061,-0.669056,-0.685977,-0.067925,0.808380,1.385235,1.455245,0.540762 ,0.139570,-1.038133,0.080121,-0.102159,-0.068675,0.515445,0.600459,0.655325,0.610604,0.482337,0.079108,-0.118951,-0.050178,0.007500,-0.200622]
        x2 = [-2.368030,-2.607095,-1.277660,0.301499,1.346982,1.885968,1.765950,1.242890,-0.464786,0.186658,-0.036450,-0.396513,-0.157115,-0.012962,0.378752,-0.151658,0.774253,0.646541,0.311877,-0.694177,-0.412918,-0.338630,0.276635]
        x3 = np.array(x1)+np.array(x2)**2 # ad hoc

        X = np.array([x1, x2, x3]).T

        result = np.array([[ 1.        ,  0.62664048,  0.62095623], \
        [ 0.62664048,  1.        ,  0.1783183 ], \
        [ 0.62095623,  0.1783183 ,  1.        ]])

        dcca_corr = topcorr.dcca(X)

        assert_array_almost_equal(dcca_corr, result)






