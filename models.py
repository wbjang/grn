import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

class GRN(nn.Module):
    """
    This is the Graph Recurrent Networks which composed of RNN, Attention and Dense Network
    Input: Rolled X - (n_iters, n_nodes, n_feats)
    Output: (n_nodes, n_class)
    """
    def __init__(self, n_iters, n_nodes, n_feats, n_hids, n_class, d1, d2, d3):
        super(GRN, self).__init__()
        self.r = R(n_nodes, n_feats, n_hids, n_iters, d1)
        self.a = attn(n_hids, d2)
        self.d = dense(n_hids, n_class, d3)

    def forward(self, X):
        out1 = self.r(X)
        att1 = self.a(out1)
        out2 = self.d(out1, att1)
        return out2
    
    def attn(self, X):
        out1 = self.r(X)
        att1 = self.a(out1)
        return att1

class R(nn.Module):
    """
    This is RNN Cell - activation function: ReLU
    Input: Tensorized rolled X - (n_iters, n_nodes, n_feats)
    Output: (n_iters, n_nodes, n_hids)
    """
    def __init__(self, n_nodes, n_feats, n_hids, n_iters, dropout):
        super(R, self).__init__()
        self.nnodes = n_nodes
        self.nfeats = n_feats
        self.nhids = n_hids
        self.niters = n_iters
        self.h = torch.zeros(n_nodes, n_hids)
        self.rnn = nn.RNNCell(n_feats, n_hids, nonlinearity='relu')
        self.init = 0
        self.dropout = dropout
        self.ln = nn.LayerNorm(n_hids)
    def forward(self, X):
        outs = torch.zeros(self.niters, self.nnodes, self.nhids).cuda()
        ### To make the overall equations similar to the convolution
        if self.init == 0: 
            self.h = torch.mm(X[0],self.rnn.weight_ih.permute(1,0)) 
            self.h += self.rnn.bias_hh + self.rnn.bias_ih
            self.init = 1
        for i in range(self.niters):
            outs[i] = self.rnn(X[i], self.h.clone())
            self.h = outs[i]
        outs = self.ln(outs)
        outs = F.dropout(outs, self.dropout, training = self.training)
        return outs

class attn(nn.Module):
    """
    The attentions are on the specific time-steps and the nodes
    For every node, the attentions are summed up to 1 over the time-steps 
    Input : Outputs from RNN - (n_iters, n_nodes, n_hids)
    Output: (n_iters, n_nodes, 1)
    """
    def __init__(self, n_hids, dropout):
        super(attn, self).__init__()
        self.attn = nn.Linear(n_hids, 1)
        self.nhids = n_hids
        self.dropout = dropout
    def forward(self, X):
        n_iters, n_nodes = X.shape[0], X.shape[1]
        X_re = X.reshape(-1, self.nhids)
        outs = self.attn(X_re)
        outs = outs.reshape(n_iters, n_nodes, 1)
        outs = F.dropout(outs, self.dropout, training = self.training)
        return F.softmax(outs, 0)

class dense(nn.Module):
    """
    This dense layer reduces the number of the dimensions of the features to
    the number of the class.
    
    First, we reduce the feature dimensions to the number of class
    Then, we multiply with the attention rate
    Finally, we sum over the n_iters

    Input : Outputs from RNN - (n_iters, n_nodes, n_hids)
    Output: (n_iters, n_nodes, n_class)
    """
    def __init__(self, n_hids, n_class, dropout):
        super(dense, self).__init__()
        self.dense = nn.Linear(n_hids, n_class)
        self.nhids = n_hids
        self.nclass = n_class
        self.dropout = dropout
    def forward(self, Y, a):
        n_nodes = Y.shape[1]
        Y_a = Y.reshape(-1, self.nhids)
        o_dense = self.dense(Y_a)
        o_dense = o_dense.reshape(-1, n_nodes, self.nclass)
        outs = torch.mul(o_dense, a)
        outs = F.dropout(outs, self.dropout, training=self.training)
        return torch.sum(outs, 0)
