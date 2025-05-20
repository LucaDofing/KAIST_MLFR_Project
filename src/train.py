# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def simulate_step(x, estimated_damping, dt=0.1):
    theta, omega = x[:, 0], x[:, 1]
    damping = estimated_damping.squeeze()

    domega = - damping * omega
    omega_next = omega + domega * dt
    theta_next = theta + omega_next * dt

    return torch.stack([theta_next, omega_next], dim=1)


def run_training(model, train_loader, test_loader, device, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0.0
        count = 0

        for data in loader:
            data = data.to(device)

            # Step 1: GNN predicts latent damping
            estimated_damping = model(data)  # [N, 1]

            # Step 2: Simulate 1 step using estimated damping
            pred_next = simulate_step(data.x, estimated_damping)  # [N, 2]

            # Step 3: Compare prediction to true next state
            loss = loss_fn(pred_next, data.x_next)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_nodes
            count += data.num_nodes

        return total_loss / count

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        if epoch % 20 == 0 or epoch == 1:
            test_loss = run_epoch(test_loader, train=False)
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    print("Training done.")

