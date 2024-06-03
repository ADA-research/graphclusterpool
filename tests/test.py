import torch

def calculate_components_torch(n_nodes: int, edges: torch.tensor):
    cluster_idx = torch.zeros(n_nodes)
    already_seen = set()

    adjacency_m = torch.zeros([n_nodes, n_nodes])
    # Fill the matrix, assuming this is a cheap operation
    for e in edges:
        adjacency_m[e[0].item(),e[1].item()] = 1
        adjacency_m[e[1].item(),e[0].item()] = 1

    for i in range(n_nodes):
        adjacency_m[i,i] = 1

    clusters = []
    for id, row in enumerate(adjacency_m):
        #print(id, row)
        if id in already_seen:
            continue
        current_clus = row.clone()
        new_iter = torch.zeros(current_clus.size())

        while not torch.all(current_clus == new_iter):
            current_clus = current_clus + new_iter
            current_clus = (current_clus > 0).int()
            new_iter = (torch.sum(adjacency_m * current_clus, dim=1) > 0).int()
        
        clusters.append(current_clus > 0)
        already_seen = already_seen.union(set(current_clus.nonzero().flatten().tolist()))

    for i, e in enumerate(clusters):
        cluster_idx[e] = i
    
    return cluster_idx

edges = torch.tensor([[0,1],[1,2],[2,3], [6,7], [7,8]])

print(calculate_components_torch(9, edges))