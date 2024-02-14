import argparse
from ModelInterface import ModelInterface
import neuralmodels as nm
import pickle
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
    parser.add_argument("--nodisplay", action=argparse.BooleanOptionalAction, help="Don't print results to the terminal and show no graphs at the end.")
    parser.add_argument("--seed", type=int, help="Seed of the run")
    parser.add_argument("--filename", type=str, help="Filename to use")
    parser.add_argument("--foldindex", type=int, help="The index of the fold to run")
    return parser

def build_model(parser: argparse) -> ModelInterface:
    data = None
    test_set_ids = None
    labels = None
    type = nm.GraphConvPoolNN
    task_type_node = True
    pooltype = ClusterPooling
    if parser.dataset == "PROTEIN":
        type = nm.GraphConvPoolNNProtein
        with open("Datasets/PROTEINS/PROTEINS_full/PROTEINS.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], prodict["node_label_tensor_tuple"])]
                labels = prodict["node_label_range"]
            
        test_set_ids = []
    elif parser.dataset == "COLLAB":
        type = nm.GraphConvPoolNNCOLLAB
        with open("Datasets/COLLAB/COLLAB.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
        test_set_ids = []
    elif parser.dataset == "REDDIT-BINARY":
        type = nm.GraphConvPoolNNRedditBinary
        with open("Datasets/REDDIT-BINARY/REDDIT-BINARY.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
        test_set_ids = []
    elif parser.dataset == "REDDIT-MULTI":
        with open("Datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
        test_set_ids = []
        
    """if parser.model.startswith("GCN"):
        type = nm.GraphConvPoolNN
    elif parser.model.startswith("GUNET"):
        type = nm.GUNET
    elif parser.model is not None:
        print("Error. model argument not recognized:", parser.model)
        sys.exit(-1)
    if parser.model.endswith("Diehl"):
        pooltype = EdgePooling
    else:
        pooltype = ClusterPooling"""
    if parser.task == "graph":
        task_type_node = False
    return nm.GCNModel(data=data, labels=labels, task_type_node=task_type_node, seed=args.seed, type=type, pooltype=pooltype)

if __name__ == "__main__":
    # Define command line arguments
    parser = parser_function()

    # Process command line arguments
    args = parser.parse_args()

    model = build_model(args)
    print("Model has been built", flush=True)
    display = args.nodisplay is None or not args.nodisplay
    file = args.filename

    model.run_folds(folds=1, display=display, contExperimentFile=file, seed=args.seed, iteration_id=args.foldindex)
