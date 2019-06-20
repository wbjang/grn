## Graph Recurrent Networks for Node Classifications

With the Cora dataset, we achieved around 83.7% average.

You can easily train the model by

> **cd grn** <br/>
> **python train.py** 

or with the hyper-parameters as 

> **python train.py --wd 1e-2 --lr 1e-2 --ps 5 --d1 0.2 --d2 0.2 --d3 0.4 --n_iters 3 --dataset "cora"**

The default hyperparameters for Cora dataset are

* n_hids : 112
* n_iters: 9
* d1, d2 : 0.2 , d3: 0.4 (Dropout rates for RNN, Attention, Dense respectively)
* lr: 1e-2 (Learning rate)
* wd: 1e-2 (Weight decay)
* ps: 5 (Patience for early-stopping)

GRN is based on the simple RNN followed by two dense layers for attention and for classification separately. It incorporates the convolutional operations in RNN framework, and achieves the comparable performances with the existing models like GCN(Graph Convolutoinal Networks) and GAT(Graph Attention Networks) for Cora, Citeseer and Pubmed dataset where the adjacency matrix is fixed. GRN also computes quite fast even though it is based on RNN model.

The data set has the interesting property - as the data points jump according to the randon-walk transition matrix, and they cluster over a few centres within 10 iterations.

![How they cluster over a few points with Cora dataset](https://github.com/wayne1123/grn/blob/master/imgs/cora-10.png)

