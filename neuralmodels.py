import numpy as np
import copy
import itertools

import sklearn

import torch
import torch.nn.functional as F
#from torch.nn import BatchNorm2d
#from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import GCNConv
#from torch_geometric.nn.pool import EdgePooling
from torch_geometric.nn import global_mean_pool
from cluster_pool import ClusterPooling

from torch_geometric.data import Data

from ModelInterface import ModelInterface

"""Architecture for protein:
* 16 batch size ?
* Does not use learning rate decay?
"""
class GraphConvPoolNNProtein(torch.nn.Module):
    archName = "GCN Pooling for PROTEIN"
    def __init__(self, node_features, task_type_node, num_classes, PoolLayer: torch.nn.Module, device):
        super().__init__()
        self.n_epochs = 1000
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = PoolLayer
        self.hid_channel = 32
        self.batch_size = 16
        self.learningrate = 0.0005
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.0
        dropout_pool=0.0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)

        self.pool1 = PoolLayer(self.hid_channel, dropout=dropout_pool)
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
        
        x = torch.sigmoid(x)
        if self.num_classes == 1: #binary
            return torch.flatten(x)
        else:
            return x

"""Architecture for REDDIT-BINARY
* 1 Batch size
* Uses learning rate decay (DIEHL) halving every 50 epochs
"""
class GraphConvPoolNNRedditBinary(torch.nn.Module):
    archName = "GCN Pooling for REDDIT-BINARY"
    def __init__(self, node_features, task_type_node, num_classes, PoolLayer: torch.nn.Module, device):
        super().__init__()
        self.n_epochs = 300
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = PoolLayer
        self.hid_channel = 64
        self.batch_size = 1
        self.learningrate = 0.001
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.0
        dropout_pool=0.0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = PoolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)

        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool2 = PoolLayer(self.hid_channel, dropout=dropout_pool)
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
    def __init__(self, node_features, task_type_node, num_classes, PoolLayer: torch.nn.Module, device):
        super().__init__()
        self.n_epochs = 250
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = PoolLayer
        self.hid_channel = 128
        self.batch_size = 1
        self.learningrate = 0.00025
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.025
        dropout_pool=0.0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = PoolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        #self.pool2 = PoolLayer(self.hid_channel, dropout=dropout_pool)
        #self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

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

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)
            # return torch.nn.functional.softmax(x, dim=1)


class GraphConvPoolNN(torch.nn.Module):
    archName = "GCN REDDIT-MULTI"
    def __init__(self, node_features, task_type_node, num_classes, PoolLayer: torch.nn.Module, device):
        super().__init__()
        self.n_epochs = 50
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = PoolLayer
        self.hid_channel = 128
        self.batch_size = 1
        self.learningrate = 0.00025
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.025
        dropout_pool=0.0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)

        self.pool1 = PoolLayer(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)

        #self.pool2 = PoolLayer(self.hid_channel, dropout=dropout_pool)
        #self.conv5 = GCNConv(self.hid_channel, self.hid_channel)

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

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)
            # return torch.nn.functional.softmax(x, dim=1)


class GUNET(torch.nn.Module):
    archName = "Graph UNET2"
    def __init__(self, features, labels, device, pType=ClusterPooling):
        super().__init__()

        self.in_channels = features
        self.out_channels = labels
        self.device = device
        self.hidden_channels = 128
        if self.out_channels == 2:
            self.out_channels = 1
        self.depth = 3#Try bigger sizes? [1, 10] range makes sense for this problem
        self.n_epochs = 500
        self.num_classes = self.out_channels
        
        self.dropoutval = 0.1
        self.pooldropoutval = 0.05
        self.dropout = torch.nn.Dropout(p=self.dropoutval)

        self.poolingType = pType

        self.show_cluster_plots = True
        self.shown = False
        self.cf1 = [[] for _ in range(self.depth)]
        #self.optim = torch.optim.Adam(self.mdl.parameters(), lr=0.00020)

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, self.hidden_channels, improved=True))
        for i in range(self.depth):
            self.pools.append(self.poolingType(self.hidden_channels, dropout=self.pooldropoutval))
            self.down_convs.append(GCNConv(self.hidden_channels, self.hidden_channels, improved=True))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            self.up_convs.append(GCNConv(self.hidden_channels*2, self.hidden_channels, improved=True))
        self.up_convs.append(GCNConv(self.hidden_channels*2+self.in_channels, self.out_channels, improved=True)) #+self.in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous(), y=data[2])
        x, edge_index = data.x, data.edge_index

        x_in = torch.clone(x)

        batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        memory = [] 
        unpool_infos = []
        for i in range(self.depth):
            x = self.down_convs[i](x, edge_index)
            if self.training:
                x = self.dropout(x)
            x = F.relu(x)
            memory.append(x.clone())
            x, edge_index, batch, unpool = self.pools[i](x, edge_index.long(), batch)
            unpool_infos.append(unpool)            

        memory[0] = torch.cat((memory[0], x_in), -1) #Concatenate the input features to the output of the first convolutional layer
        x = self.down_convs[-1](x, edge_index)

        for i in range(self.depth):
            j = self.depth - 1 - i
            x, edge_index, batch = self.pools[j].unpool(x, unpool_infos.pop())
            x = torch.cat((memory.pop(), x), -1)
            x = self.up_convs[i](x, edge_index)
            if self.training and i < self.depth - 1:
                x = self.dropout(x)
            x = F.relu(x) if i < self.depth - 1 else x
                    
        return torch.sigmoid(x).flatten()


class GCNModel(ModelInterface):
    def __init__(self, data, labels, test_set_idx, task_type_node=True, type=GraphConvPoolNN, pooltype=ClusterPooling):
        super().__init__(data, labels, test_set_idx)
        self.architecture = type
        self.pooltype = pooltype
        self.task_type_node = task_type_node
        self.clfName = self.architecture.archName
        if self.architecture == GUNET:
            self.clfName = self.clfName + "- ClusterPool"
        self.n_node_features = data[0][0].size(1)
        self.n_labels = len(labels)

    def train_model(self, replace_model=True, verbose=False):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            self.clf = self.architecture(self.n_node_features, self.task_type_node, self.n_labels, self.pooltype, self.device)
            if self.architecture == GUNET:
                self.clfName = self.clfName + " " + str(self.clf.poolingType)
            self.clf.to(self.device)

        if hasattr(self.clf, "optimizer"):
            optimizer = self.clf.optimizer
        else:
            if hasattr(self.clf, "learningrate"):
                lr = self.clf.learningrate
            else:
                lr = 0.001
            # PROTEIN
            #optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.0005)
            optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)

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
        if hasattr(self.clf, "batch_size"):
            batch_size = self.clf.batch_size
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
                
                #if index > 50:
                #    break
            
            if not self.bnry:
                y_train_labels = np.argmax(y_train_probs, axis=1)
                # print("\t\t", np.unique(y_train_labels, return_counts=True), np.unique(self.y_train, return_counts=True))
            else:
                y_train_labels = np.round(y_train_probs)
            #print("\t\t Labelling diversity:", np.unique(y_train_labels, return_counts=True), ", Actual diversity:", np.unique(self.y_train, return_counts=True))
            tot_lss = tot_lss / (index + 1)
            train_metric = sklearn.metrics.accuracy_score(self.y_train[:len(y_train_labels)], y_train_labels)
            metric_list.append(train_metric)
            
            tloss.append(tot_lss)
            if verbose or True:
                print(f"\t\tEpoch {epoch}/{self.clf.n_epochs}\t Train Accuracy: {metric_list[-1]:.4f} --- Train Loss: {tot_lss:.4f}", flush=True)
            
            if tot_lss == 0.0:
                break
            
            # From Diehl paper
            """if epoch > 0 and epoch % 50 == 0:
                for g in optimizer.param_groups:
                    if verbose or True: print(f"\n\t\t[INFO] Shrinking Learning Rate from {g['lr']} to {g['lr'] / 2}\n")
                    g['lr'] = g['lr'] / 2"""

            if True:
            #if epoch == self.clf.n_epochs:
            #if epoch % 10 == 0 and epoch > 0:
                self.clf.train(mode=False)
                val_loss = 0.0
                vlbls = []
                for data in self.valid: #For every graph in the data set
                    batch = torch.tensor([0 for _ in range(data[0].size(0))])
                    data.append(batch)
                    out = self.clf(data) #Get the labels from all the nodes in one graph 
                    val_lab = data[2]
                    if not self.bnry:
                        val_lab = torch.nn.functional.one_hot(val_lab, self.n_labels)
                    val_loss += loss_func(out, val_lab.float())
                    if type(out) == tuple:
                        out = out[0]
                    
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
                    
                if verbose or True:
                    print(f"\t\t\t\t\t\tValidation result: {valid_metric:.4f} [Accuracy] --- {val_loss:.4f} [Loss] ", flush=True)
            
        self.clf.load_state_dict(best_mod)
        self.clf.train(mode=False)
        return metric_list, tloss, vmetric_list, vloss, best_mod