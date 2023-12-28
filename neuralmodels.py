import numpy as np
import math
import copy
import time
import itertools

import sklearn

import torch
import torch.nn.functional as F
#from torch.nn import BatchNorm2d
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import EdgePooling
from torch_geometric.nn import global_mean_pool
from cluster_pool import ClusterPooling

from torch_geometric.data import Data

from ModelInterface import ModelInterface

"""The model with the architecture from Diehl"""
class GraphConvPoolNN(torch.nn.Module):
    archName = "GCN Pooling"
    def __init__(self, node_features, task_type_node, num_classes, PoolLayer: torch.nn.Module, hid_channel, device):
        super().__init__()
        self.n_epochs = 2
        self.num_classes = num_classes
        self.device = device
        self.poolLayer = PoolLayer
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.0
        dropout_pool=0.0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, hid_channel)
        #self.batchnorm1 = BatchNorm(hid_channel)
        #self.conv2 = GCNConv(hid_channel, hid_channel)
        #self.batchnorm2 = BatchNorm(hid_channel)

        self.pool1 = PoolLayer(hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(hid_channel, hid_channel)
        #self.batchnorm3 = BatchNorm(hid_channel)

        #self.conv4 = GCNConv(hid_channel, hid_channel)
        #self.batchnorm4 = BatchNorm(hid_channel)

        #self.pool2 = PoolLayer(hid_channel, dropout=dropout_pool)
        #self.conv5 = GCNConv(hid_channel, hid_channel)
        #self.batchnorm5 = BatchNorm(hid_channel)

        #self.fc1 = torch.nn.Linear(hid_channel, hid_channel)
        #self.batchnorm6 = BatchNorm(hid_channel)
        self.fc2 = torch.nn.Linear(hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        #ogsize = x.size(0)
       # x_in = torch.clone(x)

        
        x = self.conv1(x, edge_index)
        #x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        #x = self.conv2(x, edge_index)
        #x = self.batchnorm2(x)
        #x = F.relu(x)
        #x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        #x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        #x = self.conv4(x, edge_index)
        #x = self.batchnorm4(x)
        #x = F.relu(x)
        #x = self.dropout(x)

        #x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        #x = self.conv5(x, edge_index)
        #x = self.batchnorm5(x)
        #x = F.relu(x)
        #x = self.dropout(x)

        if self.task_type_node: #Dealing with node classification
            #x, edge_index, batch = self.pool2.unpool(x, unpool2)
            x, edge_index, batch = self.pool1.unpool(x, unpool1)
        else: #dealing with graph classification
            x = global_mean_pool(x, batch)
            #Try global sum pool
            #Maybe try global max pool maar dat verpest misschien de gradients

        #x = self.fc1(x)
        #x = self.batchnorm6(x)
        #x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        if self.num_classes == 1: #binary
            return torch.flatten(x)
        else:
            return x


class GUNET(torch.nn.Module):
    archName = "Graph UNET2"
    def __init__(self, features, labels, hid_channel, device, pType=ClusterPooling):
        super().__init__()

        self.in_channels = features
        self.out_channels = labels
        self.device = device
        #self.hidden_channels = 128
        self.hidden_channels = hid_channel
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
            if self.training: x = self.dropout(x)
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
            if self.training and i < self.depth - 1: x = self.dropout(x)
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
            self.clf = self.architecture(self.n_node_features, self.task_type_node, self.n_labels, self.pooltype, self.hid_channel, self.device)
            if self.architecture == GUNET:
                self.clfName = self.clfName + " " + str(self.clf.poolingType)
            self.clf.to(self.device)

        if hasattr(self.clf, "optimizer"):
            optimizer = self.clf.optimizer
        else:
            optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.0005)
            #optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.00025, weight_decay=1e-4)

        self.clf.train()
        if self.bnry:
            loss_func = torch.nn.BCELoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        metric_list = []
        tloss = []
        vloss = []
        vmetric_list = []

        batch_size = 128
        best_mod = copy.deepcopy(self.clf.state_dict())
        if verbose: print(f"Running training procedure for {self.clf.n_epochs} epochs...")
        for epoch in range(self.clf.n_epochs + 1):
            tot_lss = 0.0   
            
            y_train_probs = []
            
            listindex = list(range(1, int(len(self.train) / batch_size) + 2))
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
            tot_lss = tot_lss / len(listindex)
            # From Diehl paper
            #if epoch > 0 and epoch % 50 == 0:
            #    for g in optimizer.param_groups:
            #        g['lr'] = g['lr'] / 2

            #prec, rec, threshold =  sklearn.metrics.precision_recall_curve(self.y_train, y_train_probs)
            #f1s = (2*(prec * rec)) / ((prec + rec) + 1e-10)
            #self.threshold = threshold[np.argmax(f1s)]
            
            #y_train_labels = (y_train_probs > self.threshold).astype(int)
            if not self.bnry:
                y_train_labels = np.argmax(y_train_probs, axis=1)
            else:
                y_train_labels = np.round(y_train_probs)
            train_metric = sklearn.metrics.accuracy_score(self.y_train, y_train_labels)
            metric_list.append(train_metric)
            
            tloss.append(tot_lss)
            #print("\t\t\t Label balance:", np.count_nonzero(y_train_labels == 0), np.count_nonzero(y_train_labels == 1))
            #print("\t\t\t Actual balance:", np.count_nonzero(np.array(self.y_train) == 0), np.count_nonzero(np.array(self.y_train) == 1))
            #input()
            if verbose or True:
                print(f"\t\tEpoch {epoch} Train Accuracy: {metric_list[-1]:.4f} --- Train Loss: {tot_lss:.4f}")#--- Threshold: {self.threshold}")
            if True:
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
                    if type(out) == tuple: out = out[0]
                    
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
                #else:
                #    self.clf.load_state_dict(best_mod)
                """prec, rec, threshold =  sklearn.metrics.precision_recall_curve(self.y_valid, self.y_valid_dist)
                if not ((prec+rec) == 0).any():
                    f1s = (2*(prec * rec)) / (prec + rec)
                    valid_f1 = np.max(f1s)
                vmetric_list.append(valid_f1)
                if valid_f1 >= np.max(vmetric_list): #Best validation score thusfar
                    best_mod = copy.deepcopy(self.clf.state_dict())
                    if not ((prec+rec) == 0).any(): #Can we calculate the best threshold?
                        self.threshold = threshold[np.argmax(f1s)] #Set the threshhold to the most optimal one"""
                    
                if verbose: print(f"\t\t\tValidation Loss in Epoch {epoch}: {val_loss:.4f}")
                if verbose: print(f"\t\t\tValidation Accuracy in Epoch {epoch}: {valid_metric:.4f}")
                #print(f"\t\tValid {self.MetricName}: {valid_f1:.4f} (Best: {np.max(vmetric_list):.4f}, Thresh: {self.threshold:.4f})")


            #print(f"\t\tEpoch {epoch} Train {self.MetricName}: {train_f1:.4f}, Best: {np.max(metric_list):.4f}")
            #print(f"\t\tEpoch {epoch} Train {self.MetricName}: {np.max(metric_list):.4f}")
            
        self.clf.load_state_dict(best_mod)
        self.clf.train(mode=False)
        return metric_list, tloss, vmetric_list, vloss, best_mod