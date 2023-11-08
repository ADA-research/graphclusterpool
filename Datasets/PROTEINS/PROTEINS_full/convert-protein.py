import torch as pt
import pickle


# Conversion script for PROTEINS data set to pt tensors
# Contains 43471 nodes with 29 features
# Contains 162088 edges
# Contains 1113 graphs
# Contains 43471 Node Labels (3 types)
# Contains 1113 Binary Graph Labels

edge_list_name = "PROTEINS_full_A.txt"
node_feature_name = "PROTEINS_full_node_attributes.txt"
node_labels_name = "PROTEINS_full_node_labels.txt"
graph_labels_name = "PROTEINS_full_graph_labels.txt"
node_graph_id = "PROTEINS_full_graph_indicator.txt"

#Graph data
node_id_list = [int(row) for row in open(node_graph_id).read().splitlines() if row != ""] # Sorted list
node_id_tensor = pt.tensor(node_id_list)
node_id_ranges = []
seq_start = 0
for index in range(seq_start+1, len(node_id_list)):
    if node_id_list[seq_start] != node_id_list[index]:
        node_id_ranges.append( index-seq_start )
        seq_start = index
node_id_ranges.append(len(node_id_list)-seq_start)

edge_list = [row.split(", ") for row in open(edge_list_name).read().splitlines() if row != ""]
edge_list = [[int(row[0]), int(row[1])] for row in edge_list]

edge_tensor = pt.tensor(edge_list) # Edge tensor of 162088 edges (describing directed edges?)
edge_list_splitted = [[] for _ in range(len(node_id_ranges))]
assigned = 0
for idx, edge in enumerate(edge_list):
    g_max = 0
    for gid, g_size in enumerate(node_id_ranges):
        g_max_old = g_max
        g_max += g_size
        if edge[0] <= g_max:
            edge_list_splitted[gid].append( [edge[0]-g_max_old-1, edge[1]-g_max_old-1] ) #Here we normalize node id's for the edges to be within the Graph's node range, and in the original data, indexing starts at 1
            assigned += 1
            break
edge_tensor_tuple = [ pt.tensor(el) for el in edge_list_splitted ]

node_feature_list = [row.split(",") for row in open(node_feature_name).read().splitlines() if row != ""]
node_feature_list = [[float(val) for val in row] for row in node_feature_list]

node_tensor = pt.tensor(node_feature_list) # Node tensor of 43471 nodes with 29 features each
node_tensor_tuple = node_tensor.split(node_id_ranges)

#Labels
node_labels_list = [int(row) for row in open(node_labels_name).read().splitlines() if row != ""]
node_label_tensor = pt.tensor(node_labels_list) # 43471 Node labels of either 0, 1 or 2
node_label_tensor_tuple = node_label_tensor.split(node_id_ranges)

graph_label_list = [int(row) for row in open(graph_labels_name).read().splitlines() if row != ""]
graph_label_tensor = pt.tensor(graph_label_list) # 1113 Graph Labels of either 1 or 2 (BINARY)
graph_label_tensor = graph_label_tensor - pt.min(graph_label_tensor) # Normalize to 0 and 1

#Some statistics
edge_connectivity = edge_tensor.size(0) / sum([t.size(0) * (t.size(0)-1) for t in node_tensor_tuple])
node_edge_freq = edge_tensor.flatten().bincount()

print("# Edges per node range: [", pt.min(node_edge_freq).item(), pt.median(node_edge_freq).item(), pt.max(node_edge_freq).item(), "]" )
print("Maximum graph connectivity percentage: ", edge_connectivity*100, "%")
label_count = node_label_tensor.bincount()  
print("# Type 0 node labels: ", label_count[0].item(), "(", round(label_count[0].item()/node_label_tensor.size(0) * 100, 2), "%)")
print("# Type 1 node labels: ", label_count[1].item(), "(", round(label_count[1].item()/node_label_tensor.size(0) * 100, 2), "%)")
print("# Type 2 node labels: ", label_count[2].item(), "(", round(label_count[2].item()/node_label_tensor.size(0) * 100, 2), "%)")
print()

print("# Nodes per graph range: [", min(node_id_ranges), int(sum(node_id_ranges) / len(node_id_ranges)), max(node_id_ranges), "]")
label_count = graph_label_tensor.bincount()
print("# Type 0 graph labels: ", label_count[0].item(), "(", round(label_count[0].item()/graph_label_tensor.size(0) * 100, 2), "%)")
print("# Type 1 graph labels: ", label_count[1].item(), "(", round(label_count[1].item()/graph_label_tensor.size(0) * 100, 2), "%)")

node_labels = node_label_tensor.unique().tolist()
graph_labels = graph_label_tensor.unique().tolist()
#node_label_tensor = pt.nn.functional.one_hot(node_label_tensor) #Change to one hot encoding, as its not binary classification
#node_label_tensor_tuple = node_label_tensor.split(node_id_ranges)
data_dict = {"node_tensor_tuple": node_tensor_tuple,
             "edge_tensor_tuple": edge_tensor_tuple,
             "node_label_tensor_tuple": node_label_tensor_tuple,
             "graph_label_tensor": graph_label_tensor,
             "node_label_range": node_labels,
             "graph_label_range": graph_labels}

fname = "PROTEINS.pkl"
with open(fname, 'wb') as f:
    pickle.dump(data_dict, f)
    print("Formatted Graph data is stored in ", fname)