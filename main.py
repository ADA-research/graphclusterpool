import argparse
from ModelInterface import ModelInterface
import neuralmodels as nm
import pickle
import random
import torch
from torch_geometric.nn.pool import EdgePooling
from cluster_pool import ClusterPooling

import sys

# Define arg parse
def parser_function() -> argparse.ArgumentParser:
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(
        description=("Runs selected Graph Neural Network on specified task for specified data set"),
        )
    parser.add_argument("--model", default="GCN", type=str, help="Model type. Default: GCN. Available: GCN-Diehl, GUNET, GUNET-Diehl")
    parser.add_argument("--dataset", default="PROTEIN", type=str, help="Data set name. Default: PROTEIN. Available: ")
    parser.add_argument("--task", default="node", type=str, help="Classification task for the model. Default: node. Available: graph")
    return parser

def build_model(parser: argparse) -> ModelInterface:
    data = None
    test_set_ids = None
    labels = None
    type = nm.GraphConvPoolNN
    task_type_node = True
    if parser.dataset == "PROTEIN":
        #print(os.listdir("Datasets"))
        with open("Datasets/PROTEINS/PROTEINS_full/PROTEINS.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], prodict["node_label_tensor_tuple"])]
                labels = prodict["node_label_range"]
            
        test_set_ids = [16, 21, 22, 24, 27, 34, 80, 106, 124, 125, 127, 137, 146, 147, 153, 157, 187, 194, 195, 202, 204, 223, 246, 267, 270, 285, 290, 310, 315, 317, 334, 337, 347, 349, 389, 391, 407, 410, 421, 436, 439, 448, 467, 471, 503, 515, 531, 533, 541, 543, 546, 554, 578, 598, 605, 608, 614, 615, 619, 635, 652, 666, 676, 681, 695, 696, 714, 719, 722, 726, 745, 753, 769, 777, 780, 805, 808, 810, 812, 813, 814, 816, 835, 848, 852, 856, 882, 888, 891, 898, 902, 919, 934, 936, 946, 948, 965, 984, 992, 1016, 1017, 1019, 1024, 1033, 1061, 1067, 1079, 1089, 1094, 1099, 1112]
        test_set_ids = []
    if parser.model.startswith("GCN"):
        type = nm.GraphConvPoolNN
    elif parser.model.startswith("GUNET"):
        type = nm.GUNET
    else:
        print("Error. model argument not recognized:", parser.model)
        sys.exit(-1)
    if parser.model.endswith("Diehl"):
        pooltype = EdgePooling
    else:
        pooltype = ClusterPooling
    if parser.task == "graph":
        task_type_node = False
    return nm.GCNModel(data=data, labels=labels, task_type_node=task_type_node, test_set_idx=test_set_ids, type=type, pooltype=pooltype)

if __name__ == "__main__":
    # Define command line arguments
    parser = parser_function()

    # Process command line arguments
    args = parser.parse_args()
    

    model = build_model(args)
    print("Model has been built")
    model.run_folds(folds=10, kCross=False)