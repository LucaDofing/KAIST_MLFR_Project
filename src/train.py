# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.preprocess import AngleNormalizer


def simulate_step(x_transformed, estimated_damping, dt, normalizer):
    """
    Simulate one step using transformed features [sin(theta), cos(theta), norm_omega]
    
    Args:
        x_transformed: Tensor of shape [N, 3] with [sin(theta), cos(theta), norm_omega]
        estimated_damping: Tensor of shape [N, 1] with predicted damping coefficients
        dt: Time step size
        normalizer: AngleNormalizer instance for inverse transformations
    
    Returns:
        Tensor of shape [N, 3] with predicted next state [sin(theta_next), cos(theta_next), norm_omega_next]
    """
    # Extract transformed features
    sin_theta = x_transformed[:, 0]
    cos_theta = x_transformed[:, 1]
    norm_omega = x_transformed[:, 2]
    
    # Reconstruct theta from sin and cos (using atan2 for correct quadrant)
    theta = torch.atan2(sin_theta, cos_theta)
    
    # Denormalize omega
    omega = normalizer.inverse_transform_omega(norm_omega)
    
    # Ensure estimated_damping has the correct shape for broadcasting
    damping = estimated_damping.squeeze(-1)  # Squeeze the last dim: [N]
    
    # Apply physics model
    domega_dt = -damping * omega  # Angular acceleration
    omega_next = omega + domega_dt * dt
    theta_next = theta + omega_next * dt  # Semi-implicit Euler
    
    # Transform back to the feature space
    sin_theta_next = torch.sin(theta_next)
    cos_theta_next = torch.cos(theta_next)
    norm_omega_next = (omega_next - normalizer.omega_mean) / normalizer.omega_std
    
    return torch.stack([sin_theta_next, cos_theta_next, norm_omega_next], dim=1)


def run_training(model, train_loader, test_loader, device, normalizer, epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, lambda_supervised=0.5):
    """
    Train the model with a combination of supervised and unsupervised losses
    
    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to train on
        normalizer: AngleNormalizer instance
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        lambda_supervised: Weight for the supervised loss component (0-1)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_dynamics_loss = 0.0
        total_damping_loss = 0.0
        total_combined_loss = 0.0
        count = 0

        for data in loader:
            data = data.to(device)

            # Step 1: GNN predicts latent damping
            estimated_damping = model(data)  # [N, 1] where N is num_nodes_in_batch

            # Step 2: Calculate supervised loss (damping prediction)
            true_damping = data.y_true_damping
            damping_loss = mse_loss(estimated_damping, true_damping)
            
            # Step 3: Simulate 1 step using estimated damping (unsupervised)
            current_dt = data.dt_step[0].item()  # Get dt from the first graph in batch
            pred_next = simulate_step(data.x, estimated_damping, dt=current_dt, normalizer=normalizer)
            
            # Step 4: Calculate unsupervised loss (dynamics prediction)
            dynamics_loss = mse_loss(pred_next, data.x_next)
            
            # Step 5: Combine losses with weighting
            combined_loss = lambda_supervised * damping_loss + (1 - lambda_supervised) * dynamics_loss

            if train:
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

            # Track losses
            total_dynamics_loss += dynamics_loss.item() * data.num_nodes
            total_damping_loss += damping_loss.item() * data.num_nodes
            total_combined_loss += combined_loss.item() * data.num_nodes
            count += data.num_nodes
            
        avg_dynamics_loss = total_dynamics_loss / count
        avg_damping_loss = total_damping_loss / count
        avg_combined_loss = total_combined_loss / count
        
        return avg_combined_loss, avg_dynamics_loss, avg_damping_loss

    print("Starting training...")
    print(f"Using lambda_supervised = {lambda_supervised}")
    for epoch in range(1, epochs + 1):
        train_loss, train_dyn_loss, train_damp_loss = run_epoch(train_loader, train=True)
        if epoch % 20 == 0 or epoch == 1:
            test_loss, test_dyn_loss, test_damp_loss = run_epoch(test_loader, train=False)
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} (Dyn: {train_dyn_loss:.4f}, Damp: {train_damp_loss:.4f}) | "
                  f"Test Loss: {test_loss:.4f} (Dyn: {test_dyn_loss:.4f}, Damp: {test_damp_loss:.4f})")
    print("Training done.")


def simulate_step_physical(x_transformed, applied_torque, estimated_b, dt, mass, length_com_for_gravity, inertia_yy, gravity_accel, normalizer):
    """
    Physics-based simulation step with transformed features
    """
    # Extract transformed features
    sin_theta = x_transformed[:, 0]
    cos_theta = x_transformed[:, 1]
    norm_omega = x_transformed[:, 2]
    
    # Reconstruct theta from sin and cos
    theta = torch.atan2(sin_theta, cos_theta)
    
    # Denormalize omega
    omega = normalizer.inverse_transform_omega(norm_omega)
    
    # Extract other parameters
    b_estimated = estimated_b.squeeze(-1)
    mass_sq = mass.squeeze(-1)
    length_com_sq = length_com_for_gravity.squeeze(-1)
    inertia_yy_sq = inertia_yy.squeeze(-1)
    applied_torque_sq = applied_torque.squeeze(-1)
    g = gravity_accel.squeeze(-1)

    # Torque due to gravity (assuming theta is from vertical)
    torque_gravity = -mass_sq * g * length_com_sq * torch.sin(theta)
    
    # Torque due to damping
    torque_damping = -b_estimated * omega
    
    # Net torque
    net_torque = applied_torque_sq + torque_gravity + torque_damping
    
    # Angular acceleration
    alpha = net_torque / inertia_yy_sq

    # Update state
    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt 

    # Transform back to feature space
    sin_theta_next = torch.sin(theta_next)
    cos_theta_next = torch.cos(theta_next)
    norm_omega_next = (omega_next - normalizer.omega_mean) / normalizer.omega_std
    
    return torch.stack([sin_theta_next, cos_theta_next, norm_omega_next], dim=1)