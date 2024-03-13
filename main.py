import argparse
from ModelInterface import ModelInterface
import neuralmodels as nm
import pickle
import rerundiehl as rrd
import xugcn
from cluster_pool import ClusterPooling
from extradataxu.extra_xu_dataloader import get_extra_data

import sys
import datetime

# Define arg parse
def parser_function() -> argparse.ArgumentParser:
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(
        description=("Runs selected Graph Neural Network on specified task for specified data set"),
        )
    parser.add_argument("--dataset", default="PROTEIN", type=str, help="Data set name. Default: PROTEIN. Available: ")
    parser.add_argument("--task", default="node", type=str, help="Classification task for the model. Default: node. Available: graph")
    parser.add_argument("--nodisplay", action=argparse.BooleanOptionalAction, help="Don't print results to the terminal and show no graphs at the end.")
    parser.add_argument("--seed", type=int, help="Seed of the run")
    parser.add_argument("--filename", type=str, help="Filename to use")
    parser.add_argument("--foldindex", type=int, help="The index of the fold to run")
    parser.add_argument("--rerun", type=str, help="Which paper we want to reproduce")
    return parser

def build_model(parser: argparse) -> ModelInterface:
    data = None
    labels = None
    type = None
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
    elif parser.dataset == "REDDIT-MULTI-12K":
        type = nm.GraphConvPoolNNRedditMulti
        with open("Datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
    elif parser.dataset == "REDDIT-MULTI-5K":
        type = nm.GraphConvPoolNNRedditMulti5k
        with open("Datasets/REDDIT-MULTI-5K/REDDIT-MULTI-5K.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
    elif parser.dataset == "IMDB-BINARY":
        #This one uses reddit binary for now but should get its own
        type = nm.GraphConvPoolNNIMDBBinary
        with open("Datasets/IMDB-BINARY/IMDB-BINARY.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
    elif parser.dataset == "IMDB-MULTI":
        type = nm.GraphConvPoolNNIMDBMulti
        with open("Datasets/IMDB-MULTI/IMDB-MULTI.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
    elif parser.dataset == "NCI1":
        type = nm.GraphConvPoolNNNCI1
        with open("Datasets/NCI1/NCI1.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
    else:
        print(f"ERROR: Data set {parser.dataset} not recognized. Exiting.")
        sys.exit(-1)
    
    if parser.task == "graph":
        task_type_node = False
    if parser.rerun == "diehl":
        type = rrd.GCNDiehl
    elif parser.rerun == "xu":
        type = xugcn.GraphCNN
        extra_data = get_extra_data(parser.dataset)
        for i,e in enumerate(extra_data):
            data[i].append([e])

    return nm.GCNModel(data_name=parser.dataset, data=data, labels=labels, task_type_node=task_type_node, seed=args.seed, type=type, pooltype=pooltype)

if __name__ == "__main__":
    # Define command line arguments
    parser = parser_function()

    # Process command line arguments
    args = parser.parse_args()

    model = build_model(args)
    
    print(f"Arguments have been parsed. Starting procedure on dataset {model.dataset_name} at: {datetime.datetime.now()}", flush=True)
    display = args.nodisplay is None or not args.nodisplay
    file = args.filename

    model.run_folds(folds=1, display=display, contExperimentFile=file, seed=args.seed, iteration_id=args.foldindex)
