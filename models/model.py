import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GraphConvNet(nn.Module):
    def __init__(self) -> None:
        super(GraphConvNet, self).__init__()
        self.graph1 = GraphConv(30, 20)
        self.graph2 = GraphConv(10, 5)
        self.linear = nn.Linear(5, 2)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, g, edge_index):
        x = self.graph1(g, edge_index)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.graph2(g, x)
        x = F.leaky_relu(x)
        x = self.linear(x)
        return x