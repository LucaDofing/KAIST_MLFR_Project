import torch
from torch import nn
from torch_geometric.nn import GCNConv
from src.config import INPUT_DIM, HIDDEN_DIM

class DampingGCN(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.conv1 = GCNConv(INPUT_DIM, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)  # Output: estimated damping

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return self.linear(x)  # shape: [num_nodes, 1]
