# GCNMR
This is a TensorFlow implementation of Graph Convolution Networks with manifold regularization for semi-supervised learning, as described in our paper:
 
Kejani, M Tavassoli and Dornaika, Fadi and Talebi, H, [Graph Convolution Networks with manifold regularization for semi-supervised learning](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301362) (Neural Networks)


## Introduction

In this repo, we provide ReNode-GLCNMR's code with the Scene15 datasets as example. The graph convolution method used in this code is provided by Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017).

## Requirements
The codebase is implemented in Python 3.6.8. package versions used for development are just below
* networkx==2.2
* scipy==1.1.0
* setuptools==40.6.3
* numpy==1.15.4
* tensorflow==1.15.4
## Run the demo
```bash
cd gcn
python run_scence.py
```

## Data

There are six entries for the code.
* Feature matrix (feature.mat): An n * p sparse matrix, where n represents the number of nodes, and p represents the feature dimension of each node.
* Adjacency matrix (adj.mat): An n * n sparse matrix, where n represents the number of nodes.
* Label matrix (label.mat): An n * c matrix, where n represents the number of nodes, c represents the number of classes, and the label of the node is represented by onehot.
* Train index matrix (scence1reid.mat): An 1 * n matrix, where n represents the number of nodes.
* Validation index matrix (scence1vaid.mat): An 1 * n matrix, where n represents the number of nodes.
* Test index matrix (scence1teid.mat): An 1 * n matrix, where n represents the number of nodes.


If you want to use your own dataset, please process the data into the above state, and look at the `load_data()` function in `utils.py` for an example.


## Cite

Please cite our paper if you use this code in your own work:

```
@article{kejani2020graph,
  title={Graph Convolution Networks with manifold regularization for semi-supervised learning},
  author={Kejani, M Tavassoli and Dornaika, Fadi and Talebi, H},
  journal={Neural Networks},
  volume={127},
  pages={160--167},
  year={2020},
  publisher={Elsevier}
}
```
