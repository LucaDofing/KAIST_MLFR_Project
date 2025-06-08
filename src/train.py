# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
import torch.nn.functional as F


def simulate_step(x, estimated_damping, dt, omega_mean=0.0, omega_std=1.0): # dt is now an argument
    # Extract theta and omega from sin/cos representation and normalized omega
    sin_theta, cos_theta, omega_norm = x[:, 0], x[:, 1], x[:, 2]
    theta = torch.atan2(sin_theta, cos_theta)
    omega = omega_norm * omega_std + omega_mean  # Denormalize omega
    
    # Ensure estimated_damping has the correct shape for broadcasting if x is batched
    # x shape: [num_nodes_in_batch, 3]
    # estimated_damping shape: [num_nodes_in_batch, 1]
    damping = estimated_damping.squeeze(-1) # Squeeze the last dim: [N]

    # Semi-implicit Euler integration
    domega_dt = -damping * omega # This is an angular acceleration if I=1
    omega_next = omega + domega_dt * dt
    theta_next = theta + omega_next * dt # Uses updated omega

    # Convert back to sin/cos and normalized omega
    sin_theta_next = torch.sin(theta_next)
    cos_theta_next = torch.cos(theta_next)
    omega_next_norm = (omega_next - omega_mean) / omega_std

    return torch.stack([sin_theta_next, cos_theta_next, omega_next_norm], dim=1)


def run_training(model, train_loader, test_loader, device, epochs=200, lr=1e-2, weight_decay=1e-5):
    """Run training with improved learning rate and monitoring"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            loss = train_step(model, optimizer, data)
            train_loss += loss * data.num_nodes
            train_count += data.num_nodes
        
        # Average training loss
        train_loss = train_loss / train_count
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        for data in test_loader:
            data = data.to(device)
            loss = validate_step(model, data)
            test_loss += loss * data.num_nodes
            test_count += data.num_nodes
        
        # Average test loss
        test_loss = test_loss / test_count
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print metrics
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    print("Training done.")


def simulate_step_physical(x, applied_torque, estimated_b, dt, mass, length_com_for_gravity, inertia_yy, gravity_accel, omega_mean=0.0, omega_std=1.0):
    # Extract theta and omega from sin/cos representation and normalized omega
    sin_theta, cos_theta, omega_norm = x[:, 0], x[:, 1], x[:, 2]
    theta = torch.atan2(sin_theta, cos_theta)
    omega = omega_norm * omega_std + omega_mean  # Denormalize omega
    
    b_estimated = estimated_b.squeeze(-1)
    mass_sq = mass.squeeze(-1)
    length_com_sq = length_com_for_gravity.squeeze(-1)
    inertia_yy_sq = inertia_yy.squeeze(-1)
    applied_torque_sq = applied_torque.squeeze(-1)
    g = gravity_accel.squeeze(-1) # Should be scalar or same dim as others

    # Torque due to gravity
    torque_gravity = -mass_sq * g * length_com_sq * torch.sin(theta) # if theta is from vertical

    torque_damping = -b_estimated * omega
    
    net_torque = applied_torque_sq + torque_gravity + torque_damping
    
    alpha = net_torque / inertia_yy_sq

    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt 

    # Convert back to sin/cos and normalized omega
    sin_theta_next = torch.sin(theta_next)
    cos_theta_next = torch.cos(theta_next)
    omega_next_norm = (omega_next - omega_mean) / omega_std

    return torch.stack([sin_theta_next, cos_theta_next, omega_next_norm], dim=1)

def train_step(model, optimizer, data):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Get damping prediction
    damping = model(data)
    
    # Compute next state using full physics simulation
    next_state = model.compute_next_state(data, damping)
    
    # Only use state prediction loss
    loss = F.mse_loss(next_state, data.x_next)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()

def validate_step(model, data):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        # Get damping prediction
        damping = model(data)
        
        # Compute next state using full physics simulation
        next_state = model.compute_next_state(data, damping)
        
        # Only use state prediction loss
        loss = F.mse_loss(next_state, data.x_next)
    
    return loss.item()