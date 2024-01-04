from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
#from torch_scatter import scatter_mean
from torch_sparse import coalesce
from torch_geometric.utils import softmax

#from line_profiler import LineProfiler

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
    Edges are contracted according to their scores. Based on the selected edges,
    graph components are calculated and contracted through node coalescing.
    The node features in the components are combined by adding them together, where each
    node contributes 1/n of its features to the cluster.
    
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
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["x", "edge_index", "cluster", "batch", "new_edge_score", "old_edge_score", "selected_edges", "cluster_map", "edge_mask"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0.0,
                 add_to_edge_score=0.5):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

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
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) #Concatenates the source feature with the target features
        e = self.lin(e).view(-1) #Apply linear NN on the edge "features", view(-1) to reshape to 1 dimension
        e = F.dropout(e, p=self.dropout, training=self.training)

        e = self.compute_edge_score(e, edge_index, x.size(0)) 
        e = e + self.add_to_edge_score
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info
    
    """ New merge function for combining the nodes """
    def __merge_edges__(self, x, edge_index, batch, edge_score):
        cluster = torch.empty_like(batch, device=torch.device('cpu'))

        #We don't deal with double edged node pairs e.g. [a,b] and [b,a] in edge_index
        
        if edge_index.size(1) > x.size(0): #More edges than nodes, calculate quantile node based
            quantile = 1- (int(x.size(0) / 2) / edge_index.size(1)) #Calculate the top quantile
            edge_mask = (edge_score > (torch.quantile(edge_score, quantile)))
        else: #More nodes than edges, select half of the edges
            edge_mask = (edge_score >= torch.median(edge_score))
        
        sel_edge = edge_mask.nonzero().flatten()        
        new_edge = torch.index_select(edge_index, dim=1, index=sel_edge).to(x.device)
        
        components = calculate_components(x.size(0), new_edge) #47.3% of time
        #components = None
        #cluster, i = calculate_components_torch(x.size(0), new_edge) #
        #print(cluster)
        # Unpack the components into a cluster index for each node
        i = 0
        for c in components: #15% of time
            cluster[c] = i
            i += 1

        cluster = cluster.to(x.device)
        new_edge = new_edge.to(x.device)

        #We compute the new features as the average of the cluster's nodes' features
        new_edge_score = edge_score[sel_edge] #Get the scores that come into play
        node_reps = (x[new_edge[0]] + x[new_edge[1]]) #/2 (used to dived by two)
        node_reps = node_reps * new_edge_score.view(-1,1)
        new_x = torch.clone(x)
        
        trans_factor = torch.bincount(new_edge.flatten())
        trans_mask = (trans_factor > 0).nonzero().flatten()
        new_x[trans_mask] = 0
        trans_factor = trans_factor[trans_mask]
        
        new_x = torch.index_add(new_x, dim=0, index=new_edge[0], source=node_reps)
        new_x = torch.index_add(new_x, dim=0, index=new_edge[1], source=node_reps)
        new_x[trans_mask] = new_x[trans_mask] / trans_factor.view(-1,1) #Some nodes get index_added more than once, so divide by that number
        #new_x = scatter_mean(new_x, cluster, dim=0, dim_size=i)
        new_x = scatter_add(new_x, cluster, dim=0, dim_size=i) #This seems to work much better in terms of backprop

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N) #Remap the edges based on cluster, and coalesce removes all the doubles

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(x=x,
                                           edge_index=edge_index,
                                           cluster=cluster,
                                           batch=batch,
                                           new_edge_score=new_edge_score,
                                           old_edge_score=edge_score,
                                           selected_edges=new_edge,
                                           cluster_map=components,
                                           edge_mask=edge_mask)

        return new_x.to(x.device), new_edge_index.to(x.device), new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        REWRITE

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """
        
        # We just copy the cluster feature into every node
        # TODO: This can be done better / cleaner / more efficiently
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
