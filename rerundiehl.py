import torch
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data

from torch_geometric.nn.pool import EdgePooling

class GCNDiehl(torch.nn.Module):
    archName = "GCN Diehl"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.hid_channel = 128
        self.edge_act = EdgePooling.compute_edge_score_softmax
        if dataset_name == "PROTEIN":
            self.hid_channel = 64
        
        self.batch_size = 128
        if dataset_name == "REDDIT-BINARY" or dataset_name == "REDDIT-MULTI-12K":
            self.edge_act = EdgePooling.compute_edge_score_tanh
        #    self.batch_size = 32
        self.learningrate = 0.01
        self.lrhalving = True
        self.halvinginterval = 50
        self.optimizertype = torch.optim.Adam
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.5 #The authors say they use drop out but not what rate, so we use the default value of 0.5? As suggested by the dropout paper. set to 0.2 as i think 0.5 is too much
        dropout_pool=dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = SAGEConv(node_features, self.hid_channel)
        self.batchnorm1 = BatchNorm(self.hid_channel)

        self.conv2 = SAGEConv(self.hid_channel, self.hid_channel)
        self.batchnorm2 = BatchNorm(self.hid_channel)
        self.lin1 = torch.nn.Linear(self.hid_channel*2, self.hid_channel)

        self.pool1 = EdgePooling(self.hid_channel, dropout=dropout_pool, edge_score_method=self.edge_act)

        self.conv3 = SAGEConv(self.hid_channel, self.hid_channel)
        self.batchnorm3 = BatchNorm(self.hid_channel)

        self.conv4 = SAGEConv(self.hid_channel, self.hid_channel)
        self.batchnorm4 = BatchNorm(self.hid_channel)
        self.lin2 = torch.nn.Linear(self.hid_channel*2, self.hid_channel)

        self.pool2 = EdgePooling(self.hid_channel, dropout=dropout_pool, edge_score_method=self.edge_act)

        self.conv5 = SAGEConv(self.hid_channel, self.hid_channel)
        self.batchnorm5 = BatchNorm(self.hid_channel)

        self.conv6 = SAGEConv(self.hid_channel, self.hid_channel)
        self.batchnorm6 = BatchNorm(self.hid_channel)
        self.lin3 = torch.nn.Linear(self.hid_channel*2, self.hid_channel)

        self.pool3 = EdgePooling(self.hid_channel, dropout=dropout_pool, edge_score_method=self.edge_act)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm1(x)
        copy_x = x.clone()

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.lin1(torch.concat((x, copy_x), dim=1))
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm3(x)
        copy_x = x.clone()

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.lin2(torch.concat((x, copy_x), dim=1))
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm5(x)
        copy_x = x.clone()

        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.lin3(torch.concat((x, copy_x), dim=1))
        x = self.dropout(x)

        x, edge_index, batch, unpool3 = self.pool3(x, edge_index.long(), batch)

        x = global_mean_pool(x, batch)

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
        

"""class GCNDiehlq2(torch.nn.Module):
    archName = "GCN DIEHL"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.hid_channel = 128
        self.edge_act = EdgePooling.compute_edge_score_softmax
        if dataset_name == "PROTEIN":
            self.hid_channel = 64
        
        self.batch_size = 128
        if dataset_name == "REDDIT-BINARY" or dataset_name == "REDDIT-MULTI-12K":
            self.edge_act = EdgePooling.compute_edge_score_tanh
        #    self.batch_size = 32
        self.learningrate = 0.01
        self.lrhalving = True
        self.halvinginterval = 50
        self.optimizertype = torch.optim.Adam
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.5 #The authors say they use drop out but not what rate, so we use the default value of 0.5? As suggested by the dropout paper. set to 0.2 as i think 0.5 is too much
        dropout_pool=dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.batchnorm1 = BatchNorm(self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm2 = BatchNorm(self.hid_channel)

        self.pool1 = EdgePooling(self.hid_channel, dropout=dropout_pool, edge_score_method=self.edge_act)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm3 = BatchNorm(self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm4 = BatchNorm(self.hid_channel)

        self.pool2 = EdgePooling(self.hid_channel, dropout=dropout_pool, edge_score_method=self.edge_act)
        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm5 = BatchNorm(self.hid_channel)

        self.fc1 = torch.nn.Linear(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel, self.num_classes)

    def forward(self, data):
        batch = data[3]
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)

        x = self.conv3(x, edge_index)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)

        x = self.conv5(x, edge_index)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        if self.num_classes == 1: #binary
            x = torch.sigmoid(x)
            return torch.flatten(x)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)
            # return torch.nn.functional.softmax(x, dim=1)"""