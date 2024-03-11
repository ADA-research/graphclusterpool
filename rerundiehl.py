import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data

from torch_geometric.nn.pool import EdgePooling

class GCNDiehl(torch.nn.Module):
    archName = "GCN DIEHL"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.hid_channel = 128
        if dataset_name == "PROTEINS":
            self.hid_channel = 64
        
        self.batch_size = 128
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

        self.pool1 = EdgePooling(self.hid_channel, dropout=dropout_pool)
        self.conv3 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm3 = BatchNorm(self.hid_channel)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)
        self.batchnorm4 = BatchNorm(self.hid_channel)

        self.pool2 = EdgePooling(self.hid_channel, dropout=dropout_pool)
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
            # return torch.nn.functional.softmax(x, dim=1)