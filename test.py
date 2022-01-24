import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
def to_pyg_edgeindex(g):
    src_list = []
    dst_list = []
    attr_list = []
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i,j]>0:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(g[i,j])
    final_list = [src_list,dst_list]
    return torch.tensor(final_list)

g = torch.tensor([[0,1,0],[1,0,1],[0,1,0]])
print(g.shape)
print(to_pyg_edgeindex(g))
