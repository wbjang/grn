import os
import numpy as np
import scipy.sparse as sp

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

def split_idx(n_train, n_val, n_test):
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
    idx_shuffle = np.random.choice(n3, n3, replace=False)
    idx_train = idx_shuffle[train]
    idx_val = idx_shuffle[val]
    idx_test = idx_shuffle[test]
    return idx_train, idx_val, idx_test

