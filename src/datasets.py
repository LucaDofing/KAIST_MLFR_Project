import math, random
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.config import MAX_JOINTS


class FakePendulumDataset(InMemoryDataset):
    """
    Generates synthetic pendulum graph data.
    - mode="supervised": returns (x, y) where y = true damping
    - mode="unsupervised": returns (x, x_next), with no label
    """
    def __init__(self, num_graphs=1000, mode="supervised", transform=None):
        self.num_graphs = num_graphs
        self.mode = mode
        super().__init__('.', transform=transform)
        self.data, self.slices = self._generate()

    def _generate(self):
        if self.mode == "unsupervised":
            graphs = [self._sample_graph_unsupervised() for _ in range(self.num_graphs)]
        else:
            graphs = [self._sample_graph_supervised() for _ in range(self.num_graphs)]
        return self.collate(graphs)

    def _sample_graph_supervised(self):
        n = random.randint(1, MAX_JOINTS)
        damping = torch.empty(n).uniform_(0.05, 1.0).unsqueeze(1)

        theta = torch.empty(n).uniform_(-math.pi, math.pi)
        omega = torch.randn(n) * (1.0 - damping.squeeze()) * 3.0

        features = torch.stack([theta, omega], dim=1)

        if n == 1:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            send = torch.arange(n-1, dtype=torch.long)
            recv = send + 1
            edge_index = torch.cat(
                [torch.stack([send, recv], dim=0),
                 torch.stack([recv, send], dim=0)], dim=1)

        return Data(x=features, y=damping, edge_index=edge_index)

    def _sample_graph_unsupervised(self):
        n = random.randint(1, MAX_JOINTS)
        damping = torch.empty(n).uniform_(0.05, 1.0)

        x = []
        x_next = []

        for i in range(n):
            theta0 = random.uniform(-math.pi, math.pi)
            omega0 = random.uniform(-1.0, 1.0)
            theta1 = theta0 + omega0 * 0.1
            omega1 = omega0 - damping[i].item() * omega0 * 0.1

            x.append(torch.tensor([theta0, omega0]))
            x_next.append(torch.tensor([theta1, omega1]))

        x = torch.stack(x, dim=0)
        x_next = torch.stack(x_next, dim=0)

        if n == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            send = torch.arange(n-1, dtype=torch.long)
            recv = send + 1
            edge_index = torch.cat(
                [torch.stack([send, recv], dim=0),
                 torch.stack([recv, send], dim=0)], dim=1)

        return Data(x=x, edge_index=edge_index, x_next=x_next)
