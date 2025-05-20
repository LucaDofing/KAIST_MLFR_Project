# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def simulate_step(x, estimated_damping, dt): # dt is now an argument
    theta, omega = x[:, 0], x[:, 1]
    
    # Ensure estimated_damping has the correct shape for broadcasting if x is batched
    # x shape: [num_nodes_in_batch, 2]
    # estimated_damping shape: [num_nodes_in_batch, 1]
    damping = estimated_damping.squeeze(-1) # Squeeze the last dim: [N]

    # Simplified physics model (same as your FakePendulumDataset's implicit assumption)
    # omega_next = omega - damping * omega * dt # Explicit Euler for omega
    # theta_next = theta + omega * dt           # Explicit Euler for theta

    # OR, the model from your current train.py (semi-implicit Euler for theta)
    domega_dt = -damping * omega # This is an angular acceleration if I=1
    omega_next = omega + domega_dt * dt
    theta_next = theta + omega_next * dt # Uses updated omega

    return torch.stack([theta_next, omega_next], dim=1)


def run_training(model, train_loader, test_loader, device, epochs=200, lr=1e-3, weight_decay=1e-5): # Added lr, wd
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Use params from config
    loss_fn = torch.nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0.0
        count = 0

        for data in loader:
            data = data.to(device)

            # Step 1: GNN predicts latent damping
            estimated_damping = model(data)  # [N, 1] where N is num_nodes_in_batch

            # Step 2: Simulate 1 step using estimated damping AND correct dt
            # data.dt_step should be a tensor, ensure it's correctly broadcastable or use its value
            # If all graphs in a batch have the same dt (which they should if from same JSON or consistent across JSONs)
            # then data.dt_step might be [B] or [B,1]. We need a scalar dt for simulate_step.
            # Assuming dt_step is stored as a scalar tensor in each Data object:
            current_dt = data.dt_step[0].item() # Get dt from the first graph in batch, assume same for all

            pred_next = simulate_step(data.x, estimated_damping, dt=current_dt)

            # Step 3: Compare prediction to true next state
            loss = loss_fn(pred_next, data.x_next)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_nodes # num_nodes is total nodes in batch
            count += data.num_nodes
            
        return total_loss / count

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        if epoch % 20 == 0 or epoch == 1:
            test_loss = run_epoch(test_loader, train=False)
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    print("Training done.")