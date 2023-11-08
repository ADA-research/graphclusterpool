import argparse
from ModelInterface import ModelInterface
import neuralmodels as nm
import pickle
import random
import torch

# Define arg parse
def parser_function() -> argparse.ArgumentParser:
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(
        description=("Runs selected Graph Neural Network on specified task for specified data set"),
        )
    parser.add_argument("--model", default="GCN", type=str, help="Model type. Default: GCN. Available: GCN-Diehl, GUNET")
    parser.add_argument("--dataset", default="PROTEIN", type=str, help="Data set name. Default: PROTEIN. Available: ")
    parser.add_argument("--task", default="node", type=str, help="Classification task for the model. Default: node. Available: graph")
    return parser

def build_model(parser: argparse) -> ModelInterface:
    data = None
    test_set_ids = None
    labels = None
    type = nm.GraphConvClusterPool
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
            
        test_set_ids = random.sample(range(0, len(labels)), int((len(labels) / 100) * 15))
    if parser.model == "GCN-Diehl":
        type = nm.GraphConvPoolNN
    if parser.task == "graph":
        task_type_node = False
    return nm.GCNModel(data=data, labels=labels, task_type_node=task_type_node, test_set_idx=test_set_ids, type=type)

if __name__ == "__main__":
    # Define command line arguments
    parser = parser_function()

    # Process command line arguments
    args = parser.parse_args()
    

    model = build_model(args)
    print("Model has been built")
    model.train_model()
    print("done with training")