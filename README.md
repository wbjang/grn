## Graph Recurrent Networks for Node Classifications

With the Cora dataset, we achieved around 83.7% average.

You can easily train the model by

> **cd grn** <br/>
> **python train.py** 

or with the hyper-parameters as 

> **python train.py --wd 1e-2 --lr 1e-2 --ps 5 --d1 0.2 --d2 0.2 --d3 0.4 --n_iters 3 --dataset "cora"

Hyperparameters are
* n_hids : 112
* n_iters: 9
* d1, d2 : 0.2 , d3: 0.4 (Dropout rates for RNN, Attention, Dense respectively)
* lr: 1e-2 (Learning rate)
* wd: 1e-2 (Weight decay)
* ps: 5 (Patience for early-stopping)


