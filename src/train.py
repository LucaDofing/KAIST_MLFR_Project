# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
import torch.nn.functional as F


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
        train_metrics = {'loss': 0, 'damping_loss': 0, 'state_loss': 0}
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            metrics = train_step(model, optimizer, data)
            
            for k, v in metrics.items():
                train_metrics[k] += v * data.num_nodes
            train_count += data.num_nodes
        
        # Average training metrics
        train_metrics = {k: v/train_count for k, v in train_metrics.items()}
        
        # Validation
        model.eval()
        test_metrics = {'loss': 0, 'damping_loss': 0, 'state_loss': 0}
        test_count = 0
        
        for data in test_loader:
            data = data.to(device)
            metrics = validate_step(model, data)
            
            for k, v in metrics.items():
                test_metrics[k] += v * data.num_nodes
            test_count += data.num_nodes
        
        # Average test metrics
        test_metrics = {k: v/test_count for k, v in test_metrics.items()}
        
        # Update learning rate
        scheduler.step(test_metrics['loss'])
        
        # Early stopping
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print metrics
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(D: {train_metrics['damping_loss']:.4f}, S: {train_metrics['state_loss']:.4f}) | "
                  f"Test Loss: {test_metrics['loss']:.4f} "
                  f"(D: {test_metrics['damping_loss']:.4f}, S: {test_metrics['state_loss']:.4f})")
    
    print("Training done.")


def simulate_step_physical(x, applied_torque, estimated_b, dt, mass, length_com_for_gravity, inertia_yy, gravity_accel):
    theta, omega = x[:, 0], x[:, 1]
    b_estimated = estimated_b.squeeze(-1)
    mass_sq = mass.squeeze(-1)
    length_com_sq = length_com_for_gravity.squeeze(-1)
    inertia_yy_sq = inertia_yy.squeeze(-1)
    applied_torque_sq = applied_torque.squeeze(-1)
    g = gravity_accel.squeeze(-1) # Should be scalar or same dim as others

    # Torque due to gravity
    # Assuming theta is angle from horizontal X-axis, for a Y-axis hinge, gravity in -Z
    # Torque_gravity_about_Y = CoM_x_world * Force_z_world
    # CoM_x_world = length_com_sq * cos(theta)
    # Force_z_world = -mass_sq * g
    # torque_gravity = length_com_sq * torch.cos(theta) * (-mass_sq * g) # This seems to be the standard
    # Let's re-verify standard pendulum equation: I * alpha = -m*g*L*sin(theta_from_vertical) - b*omega
    # If your theta is angle from horizontal X-axis:
    # sin(theta_from_vertical) = cos(theta_from_horizontal)
    # So, torque_gravity = -mass_sq * g * length_com_sq * torch.cos(theta) # if theta is from horizontal
    # OR, if theta is from vertical:
    torque_gravity = -mass_sq * g * length_com_sq * torch.sin(theta) # if theta is from vertical

    torque_damping = -b_estimated * omega
    
    net_torque = applied_torque_sq + torque_gravity + torque_damping
    
    alpha = net_torque / inertia_yy_sq

    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt 

    return torch.stack([theta_next, omega_next], dim=1)

# ... run_training function (update the call to simulate_step_physical) ...
# Inside run_epoch:
# pred_next = simulate_step_physical(
#     x=data.x,
#     applied_torque=data.applied_torque_t,
#     estimated_b=estimated_b_coeff, # GNN now predicts 'b'
#     dt=current_dt,
#     mass=data.mass,
#     length_com_for_gravity=data.length_com_for_gravity,
#     inertia_yy=data.inertia_yy,
#     gravity_accel=data.gravity_accel # Pass gravity
# )

def train_step(model, optimizer, data):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Get damping prediction
    damping = model(data)  # Shape: [batch_size, 1]
    
    # Ensure target damping has the same shape
    target_damping = data.y_true_damping.unsqueeze(-1)  # Add dimension to match [batch_size, 1]
    
    # Compute next state using full physics simulation
    next_state = model.compute_next_state(data, damping)
    
    # Compute losses with proper scaling
    damping_loss = F.mse_loss(damping, target_damping)  # Now both have shape [batch_size, 1]
    state_loss = F.mse_loss(next_state, data.x_next)
    
    # Scale the losses to be of similar magnitude
    # Damping is typically small (0-1), while state values can be larger
    damping_scale = 10.0  # Increase weight of damping loss
    state_scale = 1.0
    
    # Combined loss with scaling
    loss = damping_scale * damping_loss + state_scale * state_loss
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'damping_loss': damping_loss.item(),
        'state_loss': state_loss.item()
    }

def validate_step(model, data):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        # Get damping prediction
        damping = model(data)  # Shape: [batch_size, 1]
        
        # Ensure target damping has the same shape
        target_damping = data.y_true_damping.unsqueeze(-1)  # Add dimension to match [batch_size, 1]
        
        # Compute next state using full physics simulation
        next_state = model.compute_next_state(data, damping)
        
        # Compute losses with proper scaling
        damping_loss = F.mse_loss(damping, target_damping)
        state_loss = F.mse_loss(next_state, data.x_next)
        
        # Scale the losses to be of similar magnitude
        damping_scale = 10.0
        state_scale = 1.0
        
        # Combined loss with scaling
        loss = damping_scale * damping_loss + state_scale * state_loss
    
    return {
        'loss': loss.item(),
        'damping_loss': damping_loss.item(),
        'state_loss': state_loss.item()
    }