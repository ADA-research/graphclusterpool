import torch
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data

class GCNXu(torch.nn.Module):
    archName = "GCN XU"
    def __init__(self, node_features, task_type_node, num_classes, dataset_name, device):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        self.device = device
        self.hid_channel = 64
        if dataset_name == "PROTEINS":
            self.hid_channel = 16
        if dataset_name == "NCI1":
            self.hid_channel = 32
        
        self.batch_size = 128
        if dataset_name == "PROTEINS":
            self.batch_size = 32

        self.learningrate = 0.01
        self.lrhalving = True
        self.halvinginterval = 50
        self.optimizertype = torch.optim.Adam
        
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.task_type_node = task_type_node

        dropout=0.5
        #if False: #When is it zero?
        #    dropout = 0
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv1 = GINConv(torch.nn.Linear(node_features, self.hid_channel),)
        self.batchnorm1 = BatchNorm(self.hid_channel)
        self.conv2 = GINConv(torch.nn.Linear(self.hid_channel, self.hid_channel))
        self.batchnorm2 = BatchNorm(self.hid_channel)
        self.conv3 = GINConv(torch.nn.Linear(self.hid_channel, self.hid_channel))
        self.batchnorm3 = BatchNorm(self.hid_channel)
        self.conv4 = GINConv(torch.nn.Linear(self.hid_channel, self.hid_channel))
        self.batchnorm4 = BatchNorm(self.hid_channel)
        self.conv5 = GINConv(torch.nn.Linear(self.hid_channel, self.hid_channel))
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

        x = self.conv3(x, edge_index)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.dropout(x)

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