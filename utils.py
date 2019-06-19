import os
import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh


path = 'dataset/cora/'
dataset = 'cora'


def load_cora():
    ### Load the node data
    idx_features_label = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
    idx_features_label, idx_features_label.shape
    features = sp.csr_matrix(idx_features_label[:,1:-1],dtype=np.float32)
    label_original = idx_features_label[:,-1]
    classes = set(label_original)
    classes_dict = {c:np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get,idx_features_label[:,-1])),dtype=np.int32)
    labels_ = np.array([np.where(r==1) for r in labels_onehot]).reshape(-1,)

    ### Load the link data
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
    idx = np.array(idx_features_label[:,0],dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
                    shape=(labels_onehot.shape[0],labels_onehot.shape[0]),
                    dtype=np.float32)
    adj_sp = adj_matrix(adj)
    deg_sp, deg_inv_sp = deg_matrix(adj_sp) 
    return features, labels_, adj_sp, deg_sp, deg_inv_sp

def adj_matrix(adj):
    """
    Make undirected graph / symmetric adjacency matrix using the link data
    Input: adj(link)
    Output: sparse matrix 
    """
    ret = adj
    ret = ret + ret.T.multiply(ret.T>ret) - ret.multiply(ret.T>ret)
    return ret

def deg_matrix(adj):
    """
    Make the degree matrix / and its inverse using the adjacency matrix
    Input: symmetric(adj)
    Format: cs.spr
    """
    rowsum = np.array(np.sum(adj, 1)).reshape(-1,)
    deg = np.diag(rowsum)
    deg_sp = sp.csr_matrix(deg)
    deg_inv = np.diag(np.power(rowsum, -1))
    deg_inv_sp = sp.csr_matrix(deg_inv)
    return deg_sp, deg_inv_sp

def split_idx(n_train, n_val, n_test, n_total):
    """
    Generate the train / val / test split using the randomized index
    Cora: 140 / 500 / 1000
    """
    n1 = n_train
    n2 = n_train + n_val
    n3 = n2 + n_test
    train = np.arange(n1)
    val = np.arange(n1+1, n2)
    test = np.arange(n2+1, n3)
    idx_shuffle = np.random.choice(n_total, n3, replace=False)
    idx_train = idx_shuffle[train]
    idx_val = idx_shuffle[val]
    idx_test = idx_shuffle[test]
    return idx_train, idx_val, idx_test

"""
Below three functions are from https://github.com/tkipf/gcn
I've modified load_data function.
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
  
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.astype('float32')
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels_ = np.zeros((labels.shape[0],))
    for i in range(labels.shape[1]):
      labels_[labels[:,i]==1] = np.int(i)
      
    return adj, features, labels_
