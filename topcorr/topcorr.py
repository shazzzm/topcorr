import networkx as nx
import collections
import numpy as np
from functools import partial
import operator

def _calculate_new_faces(faces, new, old_set):
    """
    Calculates the new triangular faces for the network when we 
    add in new

    Parameters
    -----------
    faces : list
        a list of the faces present in the graph
    new : int
        the node id that is being added to the face
    old_set : set
        the old face that the node is being added to

    Returns
    -------
    None
    """
    faces.remove(frozenset(old_set))

    faces.add(frozenset([new, old_set[1], old_set[0]]))
    faces.add(frozenset([new, old_set[0], old_set[2]]))
    faces.add(frozenset([new, old_set[1], old_set[2]]))

def _add_triangular_face(G, new, old_set, C, faces):
    """
    Adds a new triangle to the networkx graph G

    Parameters
    -----------
    G : networkx graph
        the networkx graph to add the new face to
    new : int
        the node id that is being added to the face
    C : array_like
        correlation matrix
    old_set : set
        the old face that the node is being added to
    faces : list
        a list of the faces present in the graph

    Returns
    -------
    None
    """
    if isinstance(new, collections.Sized):
        raise ValueError("New should be a scaler")

    if len(old_set) > 3:
        raise ValueError("Old set is not the right size!")
    for j in old_set:
        G.add_edge(new, j, weight=C[new, j])

def tmfg(corr, absolute=False, threshold_mean=True):
    """
    Constructs a TMFG from the supplied correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix
    absolute : bool
        whether to use the absolute correlation values for chooisng weights or normal ones
    threshold_mean : bool
        this will discard all correlations below the mean value when selecting the first 4 
        vertices, as in the original implementation

    Returns
    -------
    networkx graph
        The Triangular Maximally Filtered Graph
    """
    p = corr.shape[0]

    if absolute:
        weight_corr = np.abs(corr)
    else:
        weight_corr = corr

    # Find the 4 most central vertices
    new_weight = weight_corr.copy()
    if threshold_mean:
        new_weight[new_weight < new_weight.mean()] = 0
    degree_centrality = new_weight.sum(axis=0)
    ind = np.argsort(degree_centrality)[::-1]
    starters = ind[0:4]
    starters_set = set(starters)
    not_in = set(range(p))
    not_in = not_in.difference(starters_set)
    G = nx.Graph()
    G.add_nodes_from(range(p))

    # Add the tetrahedron in
    faces = set()
    _add_triangular_face(G, ind[0], set([ind[1], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[1], set([ind[2], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[0], set([ind[2], ind[3]]), corr, faces)
    _add_triangular_face(G, ind[2], set([ind[1], ind[3]]), corr, faces)

    faces.add(frozenset([ind[0], ind[1], ind[3]]))
    faces.add(frozenset( [ind[1], ind[2], ind[3]] ))
    faces.add(frozenset([ind[0], ind[2], ind[3]]))
    faces.add(frozenset([ind[0], ind[1], ind[2]]))

    while len(not_in) > 0:
        #to_check = permutations(starters_set, 3)

        max_corr = -np.inf
        max_i = -1
        nodes_correlated_with = None
        not_in_arr = np.array(list(not_in))

        # Find the node most correlated with the faces in the TMFG currently
        for ind in faces:
            ind = list(ind)
            ind_arr = np.array(ind)
            most_related = weight_corr[ind_arr, :][:, not_in_arr].sum(axis=0)
            ind_2 = np.argsort(most_related)[::-1]
            curr_corr = most_related[ind_2[0]]

            if curr_corr > max_corr:
                max_corr = curr_corr
                max_i = not_in_arr[[ind_2[0]]]
                nodes_correlated_with = ind

        starters_set = starters_set.union(set(max_i))
        not_in = not_in.difference(starters_set)
        _add_triangular_face(G, max_i[0], nodes_correlated_with, corr, faces)
        _calculate_new_faces(faces, max_i[0], nodes_correlated_with)
    
    return G

def pmfg(corr):
    """
    Constructs a PMFG from the correlation matrix specified

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

    Returns
    -------
    networkx graph
        The Planar Maximally Filtered Graph
    """
    vals = np.argsort(corr.flatten(), axis=None)[::-1]
    pmfg = nx.Graph()
    p = corr.shape[0]
    pmfg.add_nodes_from(range(p))
    for v in vals:
        idx_i, idx_j = np.unravel_index(v, (p, p))
        
        if idx_i == idx_j:
            continue

        pmfg.add_edge(idx_i, idx_j, weight=corr[idx_i, idx_j])
        if not nx.check_planarity(pmfg)[0]:
            pmfg.remove_edge(idx_i, idx_j)

        if len(pmfg.edges()) == 3 *  (p - 2):
            break

    return pmfg

def _in_same_component(components, i, j):
    """
    Checks to see if nodes i and j are in the same component in the MST

    Parameters
    -----------
    components : list 
        list containing the current components of the MST
    i : int
        integer of node i
    j : int
        integer of node j
    
    Returns
    -------
    bool
        True if i and j are in the same component, False otherwise
    """
    for c in components:
        if i in c and j in c:
            return True

    return False

def _merge_components(components, i, j):
    """
    Merges the components that contain nodes i and j

    Parameters
    -----------
    components : list 
        list containing the current components of the MST
    i : int
        integer of node i
    j : int
        integer of node j
    
    Returns
    -------
    list
        list containing the new components of the MST
    """
    c1 = None
    c2 = None
    c1_i = None
    c2_i = None
    for idx, c in enumerate(components):
        if i in c:
            c1 = c
            c1_i = idx
        
        if j in c:
            c2 = c
            c2_i = idx

    c1 |= c2

    del components[c2_i]
    return components

def mst(corr, algorithm = "kruskal"):
    """
    Constructs a minimum spanning tree from the specified correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

    algorithm : string 
        indicates which algorithm to use - currently either "prim" or "kruskal"
    Returns
    -------
    networkx graph
        The Minimum Spanning Tree
    """

    if algorithm == "prim":
        return(_prim(corr))
    else :
        return(_kruskal(corr))


def _kruskal(corr):
    """
    Constructs a minimum spanning tree using Kruskal's algorithm from the specified
    correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

        Returns
    -------
    networkx graph
        The Minimum Spanning Tree
    """
    p = corr.shape[0]
    vals = np.argsort(corr.flatten(), axis=None)[::-1]
    components = [set([x]) for x in range(p)]
    mst_G = nx.Graph()
    for v in vals:
        idx_i, idx_j = np.unravel_index(v, (p, p))
        if idx_i == idx_j:
            continue

        if _in_same_component(components, idx_i, idx_j):
            continue
        else:
            mst_G.add_edge(idx_i, idx_j, weight=corr[idx_i, idx_j])
            components = _merge_components(components, idx_i, idx_j)
        if len(mst_G.edges()) == p - 1:
            break

    return mst_G

def _prim(corr):
    """
    Constructs a minimum spanning tree using Prim's algorithm from the specified
    correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

        Returns
    -------
    networkx graph
        The Minimum Spanning Tree
    """
    p = corr.shape[0]
    mst_G = nx.Graph()

    starting_node = 0 # TODO: Make this random
    nodes_in_tree = [starting_node]
    nodes_not_in_tree = list(range(1, p))
    for i in range(p-1):
        # Find the edge with the highest correlation to a vertex
        # not already in the graph
        sub_matrix = corr[np.array(nodes_in_tree), :][:, np.array(nodes_not_in_tree)]
        largest_edge = np.argmax(sub_matrix)
        if i == 0:
            mst_G.add_edge(starting_node, largest_edge+1, weight = corr[starting_node, largest_edge+1])
            nodes_in_tree.append(largest_edge+1)
            nodes_not_in_tree.remove(largest_edge+1)
        else:
            i,j = np.unravel_index(largest_edge, sub_matrix.shape)

            # Turn i and j from local coordinates in the submatrix to ones from the whole
            # correlation matrix
            node_to_add_1 = nodes_in_tree[i]
            node_to_add_2 = nodes_not_in_tree[j]
            mst_G.add_edge(node_to_add_1, node_to_add_2, weight = corr[node_to_add_1, node_to_add_2])
            nodes_in_tree.append(node_to_add_2)
            nodes_not_in_tree.remove(node_to_add_2)

    return mst_G

def threshold(corr, threshold, binary=False, absolute=True):
    """
    Thresholds the correlation matrix at the set level

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix to threshold
    threshold : float
        threshold at which to keep values
    binary : bool, optional
        whether the nonzero values are set to 1 or left as floats (default False)
    absolute : bool, optional
        whether to threshold the plain or absolute values (default True)
    
    Returns
    -------
    array_like
        thresholded matrix
    """
    corr = corr.copy()
    if absolute:
        corr[np.abs(corr) < threshold] = 0
    else:
        corr[corr < threshold] = 0

    if binary:
        corr[np.abs(corr) > 0] = 1

    return corr

def _calculate_partial_correlation(corr, i, j, k):
    """
    Calculates the partial correlation between i and jm
    given k

    Parameters
    -----------
    corr : array_like
        correlation matrix
    i : int
        first variable
    j : int
        second variable
    k : int
        variable to remove the effect of
    """
    dem = (1 - corr[i, k]**2) * (1 - corr[k, j]**2)
    return (corr[i, j] - corr[i, k] * corr[k, j]) / np.sqrt(dem)

def dependency_network(corr):
    """
    Calculates a dependency network - see "Dominating Clasp of the Financial
    Sector Revealed by Partial Correlation Analysis of the Stock Market"
    for more details

    Parameters
    -----------
    corr : array_like
        correlation matrix

    Returns
    -------
    array_like
        dependency network adjacency matrix
    """ 
    p = corr.shape[0]
    ind = np.arange(p)
    D = np.zeros((p, p, p))
    for i in range(p):
        for j in range(p):
            for k in range(p):
                if i == j or i == k or j == k:
                    continue
                D[i, j, k] = (corr[i, j] - _calculate_partial_correlation(corr, i, j, k))
    
    for i in range(p):
        for j in range(p):
            D[i, j, j] = 1

    # Next we filter D down
    dependency_network = np.zeros((p, p))
    for i in range(p):
        for k in range(p):
            if i == k:
                continue
            dependency_network[k, i] = D[i, i!=ind, k].sum() / (p-1)

    return dependency_network

def knn(corr, k):
    """
    Calculates a k-Nearest Neighbours graph from the given correlation
    matrix - each node is allowed k edges

    Parameters
    -----------
    corr : array_like
        correlation matrix

    Returns
    -------
    Networkx Graph
        k-NN network adjacency matrix
    """ 
    p = corr.shape[0]
    G = nx.Graph()
    #ind = np.arange(p)
    for i in range(p):
        edges = corr[:, i]
        sort = np.argsort(edges)[::-1]
        num = 0
        for j in sort:
            if num == k:
                break
            if j == i:
                continue
            G.add_edge(i, j, weight=corr[i, j])
            num += 1
    return G

def partial_correlation(corr):
    """
    Calculates a partial correlation matrix from the given correlation matrix
    
    Parameters
    -----------
    corr : array_like
        correlation matrix

    Returns
    -------
    array_like
        partial correlation matrix
    """ 
    p = corr.shape[0]
    prec = np.linalg.inv(corr)
    partial_correlation = np.zeros((p, p))
    
    for i in range(p):
        for j in range(p):
            partial_correlation[i, j] = - prec[i, j] / (np.sqrt(prec[i, i] * prec[j, j]))

    np.fill_diagonal(partial_correlation, 1)

    return partial_correlation

def affinity(corr):
    """
    Calculates the affinity correlation matrix from the given correlation matrix

    Parameters
    -----------
    corr : array_like
        correlation matrix

    Returns
    -------
    array_like
        affinity correlation matrix
    """
    p = corr.shape[0]
    meta_correlation = np.zeros((p, p))
    ind = np.arange(p)
    for i in range(p):
        for j in range(i,p):
            val = np.corrcoef(corr[i, ind!=i], corr[j, ind!=i])[0, 1]
            meta_correlation[i, j] = val
            meta_correlation[j, i] = val

    A = np.multiply(meta_correlation, corr)

    return A 
 
def _redefine_matrix(Q, components, h, k, weight):
    """
    Reconstructs the correlation matrix as instructed in "SPANNING TREES AND 
    BOOTSTRAP RELIABILITY ESTIMATION IN CORRELATION-BASED NETWORKS" in equation 1

    Parameters
    -----------
    Q : array_like
        correlation matrix
    components : list
        list of sets containing the components in the MST
    h : set
        the first of the components to be merged together
    k : set
        the second of the components to be merged together
    weight : float
        the weight to put on the new edge

    Returns
    -------
    array_like
        new correlation matrix
    """
    p = Q.shape[0]
    new_Q = np.zeros((p, p))

    new_component = h.union(k)

    for i in range(p):
        for j in range(i+1, p):
            if (i in h and j in k) or (j in h and i in k):
                new_Q[i, j] = weight
                new_Q[j, i] = weight
            elif (i in new_component and j not in new_component):
                # Find component that j is in
                j_component = np.array(list(_get_component(components, j)))
                z = Q[j_component, :]
                new_val = z[:, np.array(list(new_component))].mean()
                new_Q[i, j] = new_val
                new_Q[j, i] = new_val
            else:
                new_Q[i, j] = Q[i, j]
                new_Q[j, i] = Q[i, j]

    return new_Q


def _get_component_index(components, idx_i):
    """
    Gets the index of the component that i is part of
    """
    for i,c in enumerate(components):
        if idx_i in c:
            return i

def _get_component(components, idx_i):
    """
    Gets the component that i is part of
    """
    for c in components:
        if idx_i in c:
            return c

def almst(corr):
    """
    Constructs an average linkage minimum spanning tree from the specified correlation matrix

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

    Returns
    -------
    networkx graph
        The Minimum Spanning Tree
    """
    p = corr.shape[0]
    components = [set([x]) for x in range(p)]
    mst_G = nx.Graph()
    num = 0
    Q = corr.copy()

    while num < p - 1:
        ind = np.argsort(Q, axis=None)[::-1]
        for i in ind:
            idx_i, idx_j = np.unravel_index(i, (p, p))

            if _in_same_component(components, idx_i, idx_j):
                continue
            else:
                break

        # Find out what components i and j are part of
        component_i_idx = _get_component_index(components, idx_i)
        component_j_idx = _get_component_index(components, idx_j)

        # Figure out which correlation is the largest between the components
        component_i = np.array(list(components[component_i_idx]))
        component_j = np.array(list(components[component_j_idx]))

        max_corr_idx = corr[component_i, :][:, component_j].argmax()
        idx_comp_i, idx_comp_j = np.unravel_index(max_corr_idx, (len(component_i), len(component_j)))
        max_corr_i = component_i[idx_comp_i]
        max_corr_j = component_j[idx_comp_j]

        mst_G.add_edge(max_corr_i, max_corr_j, weight=corr[max_corr_i, max_corr_j])
        
        Q = _redefine_matrix(Q, components, _get_component(components, idx_i), _get_component(components, idx_j), corr[max_corr_i, max_corr_j])
        components = _merge_components(components, max_corr_i, max_corr_j)
        num+=1

    return mst_G
    

def _mst_forest_function(D, D_orig):
    """
    Calculates the new distance matrix for each iteration of the forest procedure

    Parameters
    -----------
    D : array_like
        p x p matrix - current iteration of distance matrix

    D_orig : array_like
        p x p matrix - original distance matrix

    Returns
    -------
    array_like
        The new distance matrix
    """
    p = D.shape[0]
    D_star = np.zeros((p, p))
    indices = np.arange(p)
    for i in range(p):
        for j in range(p):
            D_star[i, j] = np.maximum(D[i, :], D_orig[:, j]).min()

    return D_star
            
                

def mst_forest(C, tol=1e-3):
    """
    Calculates the forest of MSTs as proposed by 
    "A robust filter in stock networks analysis". You may wish to 
    round your correlation matrix (C.round(n)) before putting it into this
    procedure, otherwise floating point equality means you'll just get the MST
    out.

    Parameters
    -----------
    corr : array_like
        p x p matrix - correlation matrix

    tol : float, optional
        the tolerance by which we consider two floating point numbers equal

    Returns
    -------
    networkx graph
        The resulting forest
    """
    p = C.shape[0]
    D = np.sqrt(2 - 2*C)
    D_current = D.copy()
    D_prev = D
    lim = int(1.4428 * np.log(p))
    for i in range(lim):
        D_current = _mst_forest_function(D_current, D)
        if np.isclose(D_current, D_prev, tol, 1e-5).all():
            break
        D_prev = D_current
    delta = np.zeros((p, p), dtype=int)

    diff = D_current - D
    delta[np.abs(diff) > 1e-3] = 0
    delta[np.abs(diff) < 1e-3] = 1
    np.fill_diagonal(delta, 0)

    # Construct the adjacency matrix of the new MST
    A = np.zeros((p, p))
    ind = np.nonzero(delta)

    A[ind] = C[ind]

    G = nx.from_numpy_array(A)

    return G

def dcca(X, s = 4):
    """
    Calculates the detrended cross-correlation analysis for a dataset X
    as proposed by "Detrended Cross-Correlation Analysis: A New Method for
    Analyzing Two Nonstationary Time Series" by Podobnik and Stanley

    Parameters
    -----------
    X : array_like
        n x p matrix - dataset

    s : int, optional
        Window size

    Returns
    -------
    array_like
        Correlation matrix
    """
    n, p = X.shape
    cdata = X-X.mean(axis=0)
    xx = np.cumsum(cdata,axis=0)
    indices = np.arange(s)[None, :]+np.arange(len(xx)-s+1)[:, None]
    no_runs = len(indices)
    ls_res = np.zeros((no_runs, p, s))

    for i,idx in enumerate(indices):
        xx_current = xx[idx, :]
        # Calculate the least squares fit of each variable in this time segment
        for j in range(p):
            ls_coefs = np.polyfit(np.arange(s), xx_current[:, j], deg=1)      
            
            # Calculate the residual
            ls_res[i, j, :] = xx_current[:, j] - ls_coefs[0] * np.arange(s) - ls_coefs[1]

    f_2_dcca = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            f_2_dcca[i, j] = ((ls_res[:, i, :]).flatten() * (ls_res[:, j, :].flatten())).mean()

    corr = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1, p):
            corr[i, j] = f_2_dcca[i, j]/np.sqrt(f_2_dcca[i, i] * f_2_dcca[j, j])

    corr += corr.T
    np.fill_diagonal(corr, 1)
    return corr

def change_point_detection(X, window_size=50, delta=100, num_bootstraps=50, bootstrap_size=50):
    """
    Detects the probability of a change point at each location
    """

    n, p = X.shape

    ks = np.arange(delta, n-delta)
    diffs = collections.defaultdict(partial(np.zeros, num_bootstraps))
    
    for i in range(num_bootstraps):
        ind = np.random.randint(0, n, size=bootstrap_size, dtype=int)
        for k in ks:
            ind_old = ind[ind <= k]
            ind_new = ind[ind > k]

            X_old = X[ind_old, :]
            X_new = X[ind_new, :]

            corr_old = np.corrcoef(X_old.T)
            corr_new = np.corrcoef(X_new.T)

            diffs[k][i] = np.linalg.norm(corr_new - corr_old)

    zscores = dict()
    z_b = np.zeros(num_bootstraps)
    zscores_bootstrap = collections.defaultdict(partial(np.zeros, num_bootstraps))

    for k in ks:       
        X_old = X[0:k, :]
        X_new = X[k+1:, :]

        corr_old = np.corrcoef(X_old.T)
        corr_new = np.corrcoef(X_new.T)

        diff = np.linalg.norm(corr_new - corr_old)
        zscores[k] = (diff - np.mean(diffs[k])) / np.std(diffs[k])

        for i in range(num_bootstraps):
            zscores_bootstrap[k][i] = (diffs[k][i] - np.mean(diffs[k])) / np.std(diffs[k])

    for i in range(num_bootstraps):
        max_z_b = -1
        for k in ks:
            if zscores_bootstrap[k].max() > max_z_b:
                max_z_b = zscores_bootstrap[k].max()

    max_k = max(zscores.items(), key=operator.itemgetter(1))[0]
    max_z = zscores[max_k]
    pval = (1/num_bootstraps) * np.count_nonzero(max_z_b > max_z)

    return max_k, max_z, pval, zscores

