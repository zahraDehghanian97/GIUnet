import numpy as np
import torch
from scipy.linalg import eig, eigh
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import warnings

from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import GINConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import SplineConv


import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings('ignore')
np.random.seed(0)

class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs

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
    return torch.tensor(final_list),torch.tensor(attr_list)

#real GCN
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
#
#     def forward(self, g, h):
#         print('type h:'+str(type(h)))
#         print('shape h:'+str(h.shape))
#         print(h)
#         print('type g:'+str(type(g)))
#         print('shape g:'+str(g.shape))
#         print(g)
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h

#real GAT
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.in_head = 8
#         self.out_head = 1
#         self.conv1 = GATConv(in_dim, out_dim, heads=self.in_head, dropout=p,concat=False)
#
#     def forward(self, g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#     x = self.conv1(x, edge_index)
#         #x = F.elu(x)
#         x = self.act(x)
#         return x

#real GATv2
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.in_head = 8
#         self.out_head = 1
#         self.conv1 = GATv2Conv(in_dim, out_dim, heads=self.in_head, dropout=p,concat=False)
#
#     def forward(self, g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index)
#         #x = F.elu(x)
#         x = self.act(x)
#         return x

#GIN
class GCN(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """
    def __init__(self, in_dim, out_dim, act, p):
        super(GCN,self).__init__()
        self.act = act
        self.p = p
        gin_nn = nn.Sequential(Linear_pyg(in_dim, out_dim), nn.ReLU(),Linear_pyg(out_dim, out_dim))
        self.model = GINConv(gin_nn)

    def forward(self,g,h):
        x = h
        edge_index,edge_attr = to_pyg_edgeindex(g)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.model(x,edge_index)
        x = self.act(x)
        return x

#real SAGEConv
# class GCN(nn.Module):
#     """
#     GraphSAGE Conv layer
#     """
#     def __init__(self,in_dim,out_dim,act,p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.model = SAGEConv(in_dim,out_dim,bias=True)
#
#     def forward(self,g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.model(x, edge_index)
#         x = self.act(x)
#         return x


#TransformerConv
# class GCN(nn.Module):
#     """
#     GraphSAGE Conv layer
#     """
#     def __init__(self,in_dim,out_dim,act,p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.model = TransformerConv(in_dim,out_dim,heads = 1,bias=True)
#
#     def forward(self,g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.model(x, edge_index)
#         x = self.act(x)
#         return x



class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        # return sample_k_graph_local(scores, g, h, self.k)
        # return sample_k_graph(scores, g, h, self.k)
        return top_k_graph(scores, g, h, self.k)

class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def normalized_laplacian(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """ Computes the symmetric normalized Laplacian matrix """
    num_nodes = adjacency_matrix.shape[0]
    # adjacency_matrix = adjacency_matrix - torch.eye(num_nodes, device='cuda')
    d = torch.sum(adjacency_matrix, dim=1)
    for i in range(len(d)):
        if d[i] != 0:
            d[i] = d[i] ** (-0.5)
    Dinv = torch.diag(d)
    Ln = torch.eye(len(d)) - torch.matmul(torch.matmul(Dinv, adjacency_matrix), Dinv)
    # make sure the Laplacian is symmetric
    Ln = 0.5 * (Ln + Ln.T)
    return Ln


def greedy_selection_of_samples_gpu(gl: torch.Tensor, num_of_samples: int):
    sample_set = []
    kth_power = 8

    Sc = set([i for i in range(gl.shape[0])])
    L_K = torch.matrix_power(gl, kth_power)
    L_K = 0.5 * (L_K + L_K.T)

    L_K_cpu = L_K.cpu().numpy()
    for i in range(num_of_samples):
        rc = sorted(list(Sc))
        # print('mamama',rc)
        reduced_L = L_K_cpu[rc, :]
        reduced_L = reduced_L[:, rc]
        # print('jajajaja', reduced_L)

        # u, s, _ = svd(reduced_L)
        # min_eig_index = np.argmin(s)
        # phi = np.absolute(u[min_eig_index])
        eigenvalues, eigenvectors = eigh(reduced_L)
        phi = np.absolute(eigenvectors[0])
        max_index = np.argmax(phi)
        selected = rc[max_index]
        Sc = Sc - {selected}
        sample_set.append(selected)

    return torch.tensor(sample_set, dtype=torch.long)


def sample_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]

    # 1. Compute Laplacian
    _g = g.bool().float()
    lap = normalized_laplacian(_g)
    # print(lap.shape)
    # if lap.shape[0]>300:
    #   return top_k_graph(scores, g, h, k)
    # 2. Sample
    idx = greedy_selection_of_samples_gpu(lap, max(2, int(k * num_nodes)))
    values = scores[idx]
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def spectral_clustering(adjacency_matrix: torch.Tensor, cluster_num: int):
    num_nodes = adjacency_matrix.shape[0]
    # adjacency_matrix = adjacency_matrix - torch.eye(num_nodes, device='cuda')
    adjacency_matrix = adjacency_matrix.cpu().numpy()
    adjacency_matrix = 0.5*(adjacency_matrix.T + adjacency_matrix)
    sc = SpectralClustering(cluster_num, affinity='precomputed', n_init=100, assign_labels='discretize')
    sc.fit(adjacency_matrix)
    labels = sc.labels_
    return torch.tensor(labels)


def sample_k_graph_local(scores, g, h, k):
    num_nodes = g.shape[0]
    # 1. Spectral Clustering
    new_num_nodes = max(2, int(k*num_nodes))
    labels = spectral_clustering(g, new_num_nodes)
    # print('mamama', labels)
    idx = []
    # print('yoohoo', num_nodes, new_num_nodes)
    for i in range(new_num_nodes):
        #Finding nodes of Cluster (i+1)
        cluster_list = []
        for j in range(num_nodes):
            if labels[j] == i:
                cluster_list.append(j)
        # print('lalalala', cluster_list)
        if cluster_list==[]:
          continue

        _g = g[cluster_list, :]
        _g = _g[:, cluster_list]
        #Selection of One Node From Cluster

        # 2. Compute Laplacian
        _g = _g.bool().float()
        lap = normalized_laplacian(_g)
        # print('jajajaja', lap)
        # 3. Sample
        selected_node_index = greedy_selection_of_samples_gpu(lap, 1).cpu().numpy()[0]
        selected_node = cluster_list[selected_node_index]
        idx.append(selected_node)
        # print('salam',idx)
    values = scores[idx]
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    #print(new_h.shape)
    # transfomed_h = pca(new_h, 3)
    # plot(transfomed_h)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
