import networkx as nx
import collections
import numpy as np

def _calculate_new_faces(faces, new, old_set):
    """
    Calculates the new triangular faces for the network when we 
    add in new
    """
    faces.remove(frozenset(old_set))

    faces.add(frozenset([new, old_set[1], old_set[0]]))
    faces.add(frozenset([new, old_set[0], old_set[2]]))
    faces.add(frozenset([new, old_set[1], old_set[2]]))

def _add_triangular_face(G, new, old_set, C, faces):
    """
    Adds a new triangle to the networkx graph G
    """
    if isinstance(new, collections.Sized):
        raise ValueError("New should be a scaler")

    if len(old_set) > 3:
        raise ValueError("Old set is not the right size!")
    for j in old_set:
        G.add_edge(new, j, weight=C[new, j])

def tmfg(corr):
    """
    Constructs a TMFG from the supplied correlation matrix
    """
    p = corr.shape[0]
    # Find the 4 most central vertices
    degree_centrality = corr.sum(axis=0)
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

        max_corr = -1
        max_i = -1
        nodes_correlated_with = None
        not_in_arr = np.array(list(not_in))

        # Find the node most correlated with the faces in the TMFG currently
        for ind in faces:
            ind = list(ind)
            ind_arr = np.array(ind)
            most_related = corr[ind_arr, :][:, not_in_arr].sum(axis=0)
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
    """
    for c in components:
        if i in c and j in c:
            return True

    return False

def _merge_components(components, i, j):
    """
    Merges the components that contain nodes i and j
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
    
def mst(corr):
    """
    Constructs a minimum spanning tree from the specified correlation matrix
    """
    p = corr.shape[0]
    vals = np.argsort(corr.flatten(), axis=None)[::-1]
    components = [set([x]) for x in range(p)]
    mst_G = nx.Graph()
    for v in vals:
        idx_i, idx_j = np.unravel_index(v, (p, p))
        if idx_i == idx_j:
            continue

        #idx_i, idx_j, _ = v
        if _in_same_component(components, idx_i, idx_j):
            continue
        else:
            mst_G.add_edge(idx_i, idx_j, weight=corr[idx_i, idx_j])
            components = _merge_components(components, idx_i, idx_j)
        if len(mst_G.edges()) == p - 1:
            break

    return mst_G