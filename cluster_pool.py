from collections import namedtuple
from typing import Optional

import torch
import torch.nn.functional as F
from torch_sparse import coalesce


#from line_profiler import LineProfiler
# Refine/simplify this code to specific needs so we no longer need torch_scatter
# Also look into removing dependency on torch_sparse coalesce?
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def calculate_components(n_nodes: int, edges: torch.tensor):
        # From https://stackoverflow.com/questions/10301000/python-connected-components
        # Can we do this as tensor operations that yield a cluster index per node?
        def get_all_connected_groups(graph):
            already_seen = set()
            result = []

            def get_connected_group(node, seen):
                result = []
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    #nodes = nodes or graph[node] - seen #Selects nodes unless its empty
                    nodes.update(graph[node] - already_seen)
                    result.append(node)
                return result, seen
            
            for node in graph:
                if node not in already_seen:
                    connected_group, already_seen = get_connected_group(node, already_seen)
                    result.append(connected_group)
            return result

        adj_list = {x: set() for x in range(n_nodes)} #Create an empty adjacency list for all nodes
        for edge in edges.T.tolist(): #Put values into the adjacency list  #30.9% of time
            adj_list[edge[0]].add(edge[1]) 
            adj_list[edge[1]].add(edge[0])
        
        return get_all_connected_groups(adj_list)

#@profile
def calculate_components_torch(n_nodes: int, edges: torch.tensor):
    cluster_idx = torch.zeros(n_nodes, device=edges.device, dtype=torch.int64)
    already_seen = set()
    edges = edges.T
    
    adjacency_m = torch.zeros([n_nodes, n_nodes], device=edges.device)
    # Fill the matrix, assuming this is a cheap operation
    for e in edges:
        adjacency_m[e[0].item(),e[1].item()] = 1
        adjacency_m[e[1].item(),e[0].item()] = 1

    for i in range(n_nodes):
        adjacency_m[i,i] = 1

    clusters = []
    for id, row in enumerate(adjacency_m):
        if id in already_seen:
            continue
        current_clus = row.clone()

        new_iter = torch.zeros(current_clus.size(), device=edges.device)

        while not torch.all(current_clus == new_iter):
            #print("hello")
            current_clus = current_clus + new_iter
            current_clus = (current_clus > 0).int()
            a = adjacency_m * current_clus
            b = torch.sum(a, dim=1)
            c = (b > 0)
            new_iter = c.int()
        #print()
        clusters.append(current_clus > 0)
        already_seen = already_seen.union(set(current_clus.nonzero().flatten().tolist()))
    #print(a, n_nodes, a / n_nodes)
    
    #print(len(already_seen), len(clusters), n_nodes)

    for i, e in enumerate(clusters):
        cluster_idx[e] = i
    #print(cluster_idx.nonzero().size())
    return cluster_idx, len(clusters)

class ClusterPooling(torch.nn.Module):
    r""" REWRITE THIS
    
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
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
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
                 threshold=0.0):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_tanh
        self.compute_edge_score = edge_score_method
        self.threshhold = threshold
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))

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

    def forward(self, x, edge_index, batch, directed=False):
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
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) #Concatenates the source feature with the target features
        e = self.lin(e).view(-1) #Apply linear NN on the edge "features" view(-1) to reshape to 1 dimension
        if not directed:
            e += self.lin(torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)).view(-1)
        e = e + self.bias
        e = F.dropout(e, p=self.dropout, training=self.training)

        # In case that we can treat the graph as undirected, evaluate the edge both ways
        #if not directed:
        #    e_rev = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)
        #    e_rev = self.lin(e_rev).view(-1)
        #    e_rev = F.dropout(e_rev, p=self.dropout, training=self.training)
        #  e = e + e_rev #Add the raw scores together
        #e = e + self.bias

        e = self.compute_edge_score(e)
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info
    
    """ New merge function for combining the nodes """
    def __merge_edges__(self, x, edge_index, batch, edge_score):
        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        # Select the edges from the Graph
        edge_mask = (edge_score >= self.threshhold)
        sel_edge = edge_mask.nonzero().flatten()        
        new_edge = torch.index_select(edge_index, dim=1, index=sel_edge).to(x.device)
        # Determine the components in the new graph G' = (V, E')
        components = calculate_components(x.size(0), new_edge) #47.3% of time
        # Unpack the components into a cluster index for each node
        i = 0
        for c in components: #15% of time
            cluster[c] = i
            i += 1

        cluster = cluster.to(x.device)
        new_edge = new_edge.to(x.device)

        #We compute the new features as the sum of the cluster's nodes' features, multiplied by the edge score
        new_edge_score = edge_score[sel_edge] # Get the scores of the selected edges
        # Create the nodes with the edge factor
        new_x = torch.clone(x)
        # The edge scores must be summed per node
        from torch_scatter import scatter_mul
        node_factors_left = torch.ones(new_x.size(0))
        node_factors_right = torch.ones(new_x.size(0))
        scatter_mul(new_edge_score, new_edge[0], out=node_factors_left) 
        scatter_mul(new_edge_score, new_edge[1], out=node_factors_right)
        node_factors_total = node_factors_left * node_factors_right
        new_x = (x.T * node_factors_total).T
        
        new_x = scatter_add(new_x, cluster, dim=0, dim_size=i) #This seems to work much better in terms of backprop

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N) #Remap the edges based on cluster, and coalesce removes all the doubles

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              batch=batch,
                                              cluster_map=components)

        return new_x.to(x.device), new_edge_index.to(x.device), new_batch, unpool_info

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
