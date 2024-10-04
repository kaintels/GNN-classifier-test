import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

class GraphConvNet(nn.Module):
    def __init__(self) -> None:
        super(GraphConvNet, self).__init__()
        self.graph1 = GCNConv(30, 20)
        self.graph2 = GCNConv(10, 5)
        self.linear = nn.Linear(5, 2)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.graph1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.graph2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.linear(x)
        return x