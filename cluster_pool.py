from collections import namedtuple
from typing import Optional

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_scatter import scatter_add

from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp

#from line_profiler import LineProfiler

class ClusterPooling(torch.nn.Module):
    r"""
    The cluster pooling operator from the paper `"Paper name here" <paper url here>`

    In short, a score is computed for each edge.
    Based on the selected edges, graph clusters are calculated and compressed to one
    node using an injective aggregation function (sum). Edges are remapped based on
    the node created by each cluster and the original edges.
    
    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`ClusterPooling.compute_edge_score_logsoftmax`,
            :func:`ClusterPooling.compute_edge_score_tanh`, and
            :func:`ClusterPooling.compute_edge_score_sigmoid`.
            (default: :func:`ClusterPooling.compute_edge_score_tanh`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """
    # The unpool description is rather large and could perhaps be downsized
    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "batch", "cluster_map"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0.0,
                 threshold=0.0, directed=False):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_tanh
        self.compute_edge_score = edge_score_method
        self.threshhold = threshold
        self.dropout = dropout
        self.directed = directed
        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score):
        return torch.sigmoid(raw_edge_score)

    @staticmethod
    def compute_edge_score_logsoftmax(raw_edge_score):
        return torch.nn.functional.log_softmax(raw_edge_score, dim=0)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        #First we drop the self edges as those cannot be clustered
        msk = edge_index[0] != edge_index[1]
        edge_index = edge_index[:,msk]
        if not self.directed:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)

        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) #Concatenates the source feature with the target features
        e = self.lin(e).view(-1) #Apply linear NN on the node pairs (edges) and reshape to 1 dimension
        e = F.dropout(e, p=self.dropout, training=self.training)

        e = self.compute_edge_score(e) #Non linear activation function
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info
    
    """ New merge function for combining the nodes """
    def __merge_edges__(self, X, edge_index, batch, edge_score):
        edges_contract = edge_index[..., edge_score > self.threshhold]

        adj = to_scipy_sparse_matrix(edges_contract, num_nodes=X.size(0))
        _, cluster_index = sp.csgraph.connected_components(adj, directed=True, connection="weak")

        cluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=X.device)
        C = F.one_hot(cluster_index).type(torch.float)
        A = to_dense_adj(edge_index, max_num_nodes=X.size(0)).squeeze(0)
        S = to_dense_adj(edge_index, edge_attr=edge_score, max_num_nodes=X.size(0)).squeeze(0)

        A_contract = to_dense_adj(edges_contract, max_num_nodes=X.size(0)).type(torch.int).squeeze(0)
        nodes_single = ((A_contract.sum(-1) + A_contract.sum(-2))==0).nonzero()
        S[nodes_single,nodes_single] = 1

        X_new = (S @ C).T @ X
        edge_index_new, _ = dense_to_sparse((C.T @ A @ C).fill_diagonal_(0))
        # Select the edges from the Graph
        """sel_edge = (edge_score >= self.threshhold).nonzero().flatten()        
        new_edge = torch.index_select(edge_index, dim=1, index=sel_edge).to(x.device)

        # Determine the components in the new graph G' = (V, E')
        adj = to_scipy_sparse_matrix(new_edge, num_nodes=x.size(0))
        num_clusters, components = sp.csgraph.connected_components(adj, directed=False)
        cluster = torch.tensor(components, dtype=torch.int64, device=x.device)

        new_edge_score = edge_score[sel_edge] # Get the scores of the selected edges
        #1 Create the edge score adjacency matrix
        adjacency_score = torch.zeros(x.size(0), x.size(0))
        adjacency_score[new_edge[0], new_edge[1]] = new_edge_score
        #2 reduce dimension through sum
        node_edge_score_factor = adjacency_score.sum(dim=1)
        #2B Set 1 for nodes that don't participate in any of the selected edges
        node_edge_score_factor[(node_edge_score_factor == 0).nonzero()] = 1
        #3 create new_x repr by x * summed_edge_scores
        new_x = x * node_edge_score_factor[:, None]
        #4 sum the cluster together through scatter_add
        new_x = scatter_add(new_x, cluster, dim=0, dim_size=num_clusters)


        #5 remaped the edges through coalesce
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N) #Remap the edges based on cluster, and coalesce removes all the doubles"""

        new_batch = X.new_empty(X_new.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster_index, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              batch=batch,
                                              cluster_map=cluster_index)

        return X_new.to(X.device), edge_index_new.to(X.device), new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous cluster pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`ClusterPooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """
        # We just copy the cluster feature into every node
        node_maps = unpool_info.cluster_map
        n_nodes = 0
        for c in node_maps:
            node_maps += len(c)
        import numpy as np
        repack = np.array([-1 for _ in range(n_nodes)])
        for i,c in enumerate(node_maps):
            repack[c] = i
        new_x = x[repack]

        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
