import torch
from torch import nn
from torch_geometric.nn import GCNConv

class DampingGCN(nn.Module):
    def __init__(self, in_feats=2, hidden=64, out_feats=1, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_feats, hidden))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_feats))
        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = self.act(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)  # no activation after final layer
        return x