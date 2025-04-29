import math, random
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.config import MAX_JOINTS


class FakePendulumDataset(InMemoryDataset):
    """
    Generates simple pendulum graphs:
    Each node = [angle, angular velocity], with a ground-truth damping label.
    """
    def __init__(self, num_graphs=1000, transform=None):
        self.num_graphs = num_graphs
        super().__init__('.', transform=transform)
        self.data, self.slices = self._generate()

    def _sample_graph(self):
        n = random.randint(1, MAX_JOINTS)  # 1–4 joints
        damping = torch.empty(n).uniform_(0.05, 1.0).unsqueeze(1)  # [n,1]
    
        # simulate angles
        theta = torch.empty(n).uniform_(-math.pi, math.pi)

        # simulate velocities: high damping → low velocities -----------> still super random but at least some correlation
        omega = torch.randn(n) * (1.0 - damping.squeeze()) * 3.0

        features = torch.stack([theta, omega], dim=1)   # [n,2]

        # Edge list
        if n == 1:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            send = torch.arange(n-1, dtype=torch.long)
            recv = send + 1
            edge_index = torch.cat(
                [torch.stack([send, recv], dim=0),
                torch.stack([recv, send], dim=0)], dim=1)

        return Data(x=features, y=damping, edge_index=edge_index)


    def _generate(self):
        graphs = [self._sample_graph() for _ in range(self.num_graphs)]
        return self.collate(graphs)