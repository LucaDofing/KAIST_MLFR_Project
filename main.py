# main.py

# ========== 0. Imports ==========
import math, random
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv

# ========== 1. Fake Pendulum Dataset ==========
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
        n = random.randint(1, 4)  # 1–4 joints
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

# ========== 2. Data split ==========
full_dataset = FakePendulumDataset(1200)
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [1000, 200], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 3. GNN Model ==========
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

model = DampingGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# ========== 4. Training Loop ==========
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_nodes = 0.0, 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = loss_fn(pred, data.y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
    return total_loss / total_nodes

print("Starting training...")
for epoch in range(1, 201):
    train_loss = run_epoch(train_loader, train=True)
    if epoch % 20 == 0 or epoch == 1:
        test_loss = run_epoch(test_loader, train=False)
        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

print("Training done.")

# ========== 5. Inference Demo ==========
print("\nRunning a demo prediction...")
model.eval()
sample = FakePendulumDataset(1)[0].to(device)
with torch.no_grad():
    pred_damping = model(sample)

print("True damping coefficients:", sample.y.squeeze().cpu().numpy())
print("Predicted damping coefficients:", pred_damping.squeeze().cpu().numpy())
