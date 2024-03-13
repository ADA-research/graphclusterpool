import numpy as np
import copy
import itertools

import sklearn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from cluster_pool import ClusterPooling

from torch_geometric.data import Data

from ModelInterface import ModelInterface

import xugcn

"""Notes
* Batch size does not seem to work well tried many sizes (4,8,16,32,64)
* Hidden channel seems to be optimal at 64, maybe 32 could also be good
* Learning rate seems stable
* Dropout can be 0.0, 0.05 or 0.1 but higher (0.4-0.5) also shows some potential
* Activation function for pooling works best with softmax (tried softmax, logsoftmax, sigmoid and tanh)
* Architecture is the smallest possible in number of layers to be able to memorise the data set
* The network is really good at overfitting, not so good at regularization. This is possibly due to the small data set size.
"""
class GraphConvPoolNNProtein(torch.nn.Module):
    archName = "GCN Pooling for PROTEIN"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 300
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = ClusterPooling
        self.hid_channel = 32
        self.batch_size = 1
        self.learningrate = 0.001
        self.weight_decay = 0
        self.lrhalving = True
        self.halvinginterval = 100
        dropout=0.4
        dropout_pool=dropout
        self.task_type_node = task_type_node
        if self.num_classes == 2: #binary
            self.num_classes = 1
        
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool, edge_score_method=ClusterPooling.compute_edge_score_sigmoid, threshold=0.5)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]

        data = Data(x=data[0], edge_index=data[1].t().contiguous())
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc2(x)
                
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)

class GraphConvPoolNNRedditBinary(torch.nn.Module):
    archName = "GCN Pooling for REDDIT-BINARY"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.task_type_node = task_type_node
        self.poolLayer = ClusterPooling
        self.hid_channel = 128
        self.batch_size = 1
        self.learningrate = 0.001
        self.lrhalving = True
        self.halvinginterval = 50
        dropout=0.0
        dropout_pool=dropout

        if self.num_classes == 2: #binary
            self.num_classes = 1
        
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)

        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool2 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        if self.num_classes == 1: #binary
            return torch.flatten(x)
        else:
            return x

class GraphConvPoolNNCOLLAB(torch.nn.Module):
    archName = "GCN Pooling COLLAB"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 400
        self.num_classes = num_classes
        self.device = device
        self.task_type_node = task_type_node
        self.poolLayer = ClusterPooling
        self.hid_channel = 128
        self.batch_size = 1
        #self.learningrate = 0.00025
        self.learningrate = 0.0005
        self.weight_decay = 0
        self.lrcosine = False
        self.lrhalving = True
        self.halvinginterval = 175
        dropout=0.0
        dropout_pool=0.0
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        #self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        #sself.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        #self.pool2 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        #self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

        #self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        """x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)"""

        if self.task_type_node: #Dealing with node classification
            #x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.dropout(x)
        
        x = self.fc2(x)
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)
            # return torch.nn.functional.softmax(x, dim=1)

class GraphConvPoolNNRedditMulti(torch.nn.Module):
    archName = "GCN REDDIT-MULTI"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = ClusterPooling
        self.hid_channel = 256
        self.batch_size = 1
        self.learningrate = 0.00025
        self.lrhalving = True
        self.halvinginterval = 55
        dropout=0.025
        dropout_pool=dropout
        self.task_type_node = task_type_node
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool2 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)
        
class GraphConvPoolNNRedditMulti5k(torch.nn.Module):
    archName = "GCN REDDIT-MULTI-5k"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 300
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = ClusterPooling
        self.hid_channel = 256
        self.batch_size = 1
        self.learningrate = 0.0007
        self.lrhalving = True
        self.halvinginterval = 80
        dropout=0.0
        dropout_pool=dropout
        self.task_type_node = task_type_node
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)
        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)

        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)
        self.pool2 = self.poolLayer(self.hid_channel, dropout=dropout_pool)

        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        #x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)

class GraphConvPoolNNIMDBBinary(torch.nn.Module):
    archName = "GCN Pooling for IMDB-Binary"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 100
        self.num_classes = num_classes
        self.device = device
        self.task_type_node = task_type_node
        self.poolLayer = ClusterPooling
        self.hid_channel = 32
        self.batch_size = 1
        self.learningrate = 0.0001
        self.lrcosine = False
        self.lrhalving = True
        self.halvinginterval = 22
        dropout=0.1
        dropout_pool=dropout

        if self.num_classes == 2: #binary
            self.num_classes = 1
        
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            #x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        if self.num_classes == 1: #binary
            return torch.flatten(x)
        else:
            return x

class GraphConvPoolNNIMDBMulti(torch.nn.Module):
    archName = "GCN IMDB-Multi"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 100
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = ClusterPooling
        self.hid_channel = 256
        self.batch_size = 1
        self.learningrate = 0.001
        self.lrhalving = True
        self.halvinginterval = 45
        dropout=0.00
        dropout_pool=dropout
        self.task_type_node = task_type_node
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            #x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        x = self.fc2(x)
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)

class GraphConvPoolNNNCI1(torch.nn.Module):
    archName = "GCN NCI1"
    def __init__(self, node_features, task_type_node, num_classes, datset_name, device):
        super().__init__()
        self.n_epochs = 400
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = ClusterPooling
        self.hid_channel = 128
        self.batch_size = 1
        self.learningrate = 0.001
        self.lrhalving = True
        self.halvinginterval = 80

        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.15
        dropout_pool=dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv1 = GCNConv(node_features, self.hid_channel)
        
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)
        
        self.pool1 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)
        
        self.pool2 = self.poolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        
        return torch.flatten(x)


class GCNModel(ModelInterface):
    def __init__(self, data_name, data, labels, seed=None, task_type_node=True, type=torch.nn.Module, pooltype=ClusterPooling):
        super().__init__(data_name, data, labels, seed)
        self.data_name = data_name
        self.architecture = type
        self.pooltype = pooltype
        self.task_type_node = task_type_node
        self.clfName = self.architecture.archName
        self.n_node_features = data[0][0].size(1)
        self.n_labels = len(labels)

    def train_model(self, replace_model=True, verbose=False):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            if self.architecture == xugcn.GraphCNN:
                hidden_dim = 64
                if self.dataset_name == "PROTEIN":
                    hidden_dim = 16
                if self.dataset_name == "NCI1":
                    hidden_dim = 32
                output_dim = self.n_labels
                if self.n_labels == 2:
                    output_dim = 1
                readout = "sum"
                if self.dataset_name != "PROTEIN" and  self.dataset_name != "NCI1":
                    readout = "average"
                self.clf = xugcn.GraphCNN(num_layers=5, num_mlp_layers=2, input_dim=self.n_node_features, hidden_dim=hidden_dim, output_dim=output_dim, final_dropout=0.5, learn_eps=False, graph_pooling_type=readout, neighbor_pooling_type="sum", device=self.device)
                self.clf.batch_size = 128
                if self.dataset_name == "PROTEIN" or self.dataset_name == "REDDIT-BINARY":
                    self.clf.batch_size = 32

                self.clf.learningrate = 0.01
                self.clf.lrhalving = True
                self.clf.halvinginterval = 50
                self.clf.optimizertype = torch.optim.Adam
                
            else:
                self.clf = self.architecture(self.n_node_features, self.task_type_node, self.n_labels, self.pooltype, self.device)
            self.clf.to(self.device)
            param_count = np.sum([params.size()[0] for params in self.clf.parameters()])
            print(f"Created model with {param_count} parameters.")
        if hasattr(self.clf, "optimizertype"):
            optimizer = self.clf.optimizertype(self.clf.parameters(), lr=self.clf.learningrate)
        else:
            lr = 0.001
            if hasattr(self.clf, "learningrate"):
                lr = self.clf.learningrate    
            weight_decay = 0.01
            #weight_decay = 0
            if hasattr(self.clf, "weight_decay"):
                weight_decay = self.clf.weight_decay
            #optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer = torch.optim.AdamW(self.clf.parameters(), lr=lr, weight_decay=weight_decay)

        self.clf.train()
        if self.bnry:
            loss_func = torch.nn.BCELoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        metric_list = []
        tloss = []
        vloss = []
        vmetric_list = []

        batch_size = 1
        lrhalving = False
        halvinginterval = 50
        if hasattr(self.clf, "batch_size"):
            batch_size = self.clf.batch_size
        if hasattr(self.clf, "lrhalving"):
            lrhalving = self.clf.lrhalving
            if hasattr(self.clf, "halvinginterval"):
                halvinginterval = self.clf.halvinginterval
        if hasattr(self.clf, "lrcosine") and self.clf.lrcosine:
            #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.clf.n_epochs + 1)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*(self.clf.n_epochs + 1))

        best_mod = copy.deepcopy(self.clf.state_dict())
        for epoch in range(1, self.clf.n_epochs + 1):
            tot_lss = 0.0   
            
            y_train_probs = []
            
            listindex = list(range(1, int(len(self.train) / batch_size) + 2))
            index = 0
            for index in listindex:
                starti = (index-1)*batch_size
                endi = min(index*batch_size, len(self.train))
                if starti >= len(self.train):
                    break

                nodes = [self.train[i][0] for i in range(starti, endi)]
                edges = [self.train[i][1] for i in range(starti, endi)]
                labels = [self.train[i][2] for i in range(starti, endi)]
                batchlist = list(itertools.chain.from_iterable(itertools.repeat(i, c.size(0)) for i, c in enumerate(nodes)))
                batchtensor = torch.tensor(batchlist, device=self.device)
                data = [torch.cat(nodes), torch.cat(edges), torch.cat(labels), batchtensor]
                if self.architecture == xugcn.GraphCNN:
                    extra_data = []
                    extra_data.extend([self.train[i][3][0] for i in range(starti, endi)])
                    data.append(extra_data)
                optimizer.zero_grad()
                out = self.clf(data)
                
                class_lbls = data[2]
                if not self.bnry:
                    class_lbls = torch.nn.functional.one_hot(class_lbls, self.n_labels)
                loss = loss_func(out, class_lbls.float()) #Now get the loss based on these outputs and the actual labels of the graph
                y_train_probs.extend(out.cpu().detach().numpy().tolist())
                tot_lss += loss.item()

                loss.backward()              
                optimizer.step()
            
            if not self.bnry:
                y_train_labels = np.argmax(y_train_probs, axis=1)
            else:
                y_train_labels = np.round(y_train_probs)

            tot_lss = tot_lss / (index + 1)
            train_metric = sklearn.metrics.accuracy_score(self.y_train[:len(y_train_labels)], y_train_labels)
            metric_list.append(train_metric)
            
            tloss.append(tot_lss)
            if verbose or True:
                print(f"\t\tEpoch {epoch}/{self.clf.n_epochs}\t Train Accuracy: {metric_list[-1]:.4f} --- Train Loss: {tot_lss:.4f}", flush=True)
            
            if tot_lss == 0.0:
                break
            
            # From Diehl paper
            if lrhalving and epoch > 0:
                if epoch % halvinginterval == 0:
                    for g in optimizer.param_groups:
                        if verbose or True:
                            print(f"\n\t\t[INFO] Shrinking Learning Rate from {g['lr']} to {g['lr'] / 2}\n")
                        g['lr'] = g['lr'] / 2

            if len(self.valid) > 0:
                self.clf.train(mode=False)
                val_loss = 0.0
                vlbls = []
                for data in self.valid: #For every graph in the data set
                    batch = torch.tensor([0 for _ in range(data[0].size(0))])
                    if not self.architecture == xugcn.GraphCNN:
                        data.append(batch)
                    out = self.clf(data) #Get the labels from all the nodes in one graph 
                    val_lab = data[2]
                    if not self.bnry:
                        val_lab = torch.nn.functional.one_hot(val_lab, self.n_labels)
                    val_loss += loss_func(out, val_lab.float())
                    
                    if not self.bnry:
                        out = out.argmax(dim=1)
                    vlbls.extend(np.round(out.detach().numpy()).tolist())
                
                self.clf.train()
                val_loss = val_loss.item() / len(self.valid)
                vloss.append(val_loss)
                valid_metric = sklearn.metrics.accuracy_score(self.y_valid, vlbls)
                vmetric_list.append(valid_metric)
                if valid_metric >= np.max(vmetric_list): #Best validation score thusfar
                    best_mod = copy.deepcopy(self.clf.state_dict())
                # This seems to be better (Empirically checked on PROTEINS) but would require a re-run of all experiments
                #if val_loss <= np.min(vloss): #Best validation score thusfar
                #    best_mod = copy.deepcopy(self.clf.state_dict())
                    
                if verbose or True:
                    print(f"\t\t\t\t\t\t Validation result: {valid_metric:.4f} [Accuracy] --- {val_loss:.4f} [Loss] ", flush=True)
            
        self.clf.load_state_dict(best_mod)
        self.clf.train(mode=False)
        return metric_list, tloss, vmetric_list, vloss, best_mod