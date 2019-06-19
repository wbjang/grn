import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from utils import load_cora, adj_matrix, deg_matrix, split_idx
from models import GRN
import numpy as np
import matplotlib.pyplot as plt


def unroll(X, P, n_iters):
    """
    Making n_iters number of transitions from X with P
    -> X, PX, P^2X, ... P^{n_iters-1}X
    Input: X, P, n_iters
    Output: Tensor - (n_iters, n_nodes, n_feats)
    """
    n_nodes, n_feats = X.shape[0], X.shape[1]
    outs = torch.zeros(n_iters, n_nodes, n_feats)
    if torch.cuda.is_available():
        outs = outs.cuda()
    outs[0] = X
    for i in range(1, n_iters):
        outs[i] = torch.mm(P, outs[i-1])
    return outs

def loss_ce(score, y): ### Loss - Crossentropy
    loss = nn.CrossEntropyLoss()
    return loss(score, y)

def accuracy(output, labels): # From GCN pytorch code - https://github.com/tkipf/pygcn
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test(model, X, labels, idx_test_):
    """
    Input: model, X, idx_test_(index for test data)
    Output: Loss, Accuracy
    Print out the loss / accuracy 
    """
    model.eval()
    outs = model(X)
    loss_test = loss_ce(outs[idx_test_], labels[idx_test_])
    acc_test = accuracy(outs[idx_test_], labels[idx_test_])
    print("Loss = {:.4f}".format(loss_test.item()),
          "Accuracy = {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item() 

def draw(l_train, l_val, acc_val):
    """
    Draw the training / validataion loss function and accuracy curve
    """
    fig, axes = plt.subplots(2,1)
    plt.subplot(2,1,1)
    plt.title("Loss")
    plt.plot(l_train)
    plt.plot(l_val)
    plt.legend(["Train","Val"])
    plt.subplot(2,1,2)
    plt.title("Accurcy")
    plt.plot(acc_val)
    fig.tight_layout()
    plt.show()
    

def train(model, n_iters, n_hids, n_epochs, features, P, labels, lr_, w_, p, idx_train_, idx_val_, earlystop=True):
    """
    Train function
    Input: Model, n_iters, n_hids, n_epochs, lr_, w_, run, idx_train_, idx_val_, idx_test_, p
    Output: train_loss, val_loss, val_accuracy, test_loss, test_accuracy [all : list]
    Print the loss / accuracy after finishing the training
    """
    X = unroll(features, P, n_iters)
    g_op = optim.Adam(model.parameters(), lr=lr_, weight_decay = w_)
    loss_list_train = []
    loss_list_val = [] 
    acc_list_val = []
    es = earlystopping(patience=p)
    for epoch in range(n_epochs):
        g_op.zero_grad()
        model.train()
        outs = model(X)
        loss_train = loss_ce(outs[idx_train_], labels[idx_train_])
        loss_train.backward(retain_graph=True)
        loss_list_train.append(loss_train.item())
        g_op.step()
        
        model.eval()
        outs = model(X)
        loss_val = loss_ce(outs[idx_val_], labels[idx_val_])
        acc_val = accuracy(outs[idx_val_], labels[idx_val_])
        loss_list_val.append(loss_val.item())
        acc_list_val.append(acc_val.item())
        if earlystop:
            stop = es.test(loss_list_val)
            if stop:
                break
    #t_loss, t_acc = test(model, X, idx_test_)
    return loss_list_train, loss_list_val, acc_list_val


class earlystopping():
    """
    Early-stopping 
    If the validation loss is above the best loss by the number of the patience,
    the model stops training.
    """
    def __init__(self, patience = 5):
        self.best = 1000
        self.patience = patience
        self.t = 0
    def test(self, l_list):
        current = l_list[-1]
        if current < self.best:
            self.best = current
            return False
        else:
            self.t += 1
            if self.t > self.patience:
                return True
            else:
                return False


"""
features_, labels_, adj, deg, deg_inv = load_cora()
P = torch.from_numpy(deg_inv.dot(adj.todense()))
features = torch.from_numpy(features_.todense())
labels = torch.from_numpy(labels_)
n_nodes, n_feats = features_.shape[0], features_.shape[1]
n_class = np.max(labels_) + 1
### Belows are the hyperparameters
n_hids = 112
n_iters = 7
d1 = 0.2 # Dropout rate for RNN
d2 = 0.2 # Dropout rate for attention
d3 = 0.4 # Dropout rate for dense(classification)
n_epochs = 200
lr = 1e-2 # Learning rate for the parameters
wd = 1e-2 # Weight decay for the parameters
ps = 5 #Patience rate for Early Stopping

### Making the Model
grn = GRN(n_iters, n_nodes, n_feats, n_hids, n_class, d1, d2, d3)

### If you have GPU,
if torch.cuda.is_available():
    P = P.cuda()
    features = features.cuda()
    labels = labels.cuda()
    grn = grn.cuda()

### Get the train / val / test split
idx_train_, idx_val_, idx_test_ = split_idx(140, 500, 1000)

### Train the model
l_train, l_val, acc_val = train(grn, n_iters, n_hids, n_epochs, lr, wd, ps ,idx_train_, idx_val_, idx_test_)

### Draw the loss / accuracy
draw(l_train, l_val, acc_val)

"""
