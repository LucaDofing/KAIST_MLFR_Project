# src/train.py

import torch
import torch.nn.functional as F

# This function is now defined here and used by train_step/validate_step
def simulate_step_physical(x, applied_torque, estimated_b, dt, mass, length_com_for_gravity, inertia_yy, gravity_accel):
    theta, omega = x[:, 0], x[:, 1]
    b_estimated = estimated_b.squeeze(-1)
    mass_sq = mass.squeeze(-1)
    length_com_sq = length_com_for_gravity.squeeze(-1)
    inertia_yy_sq = inertia_yy.squeeze(-1)
    applied_torque_sq = applied_torque.squeeze(-1)
    g = gravity_accel.squeeze(-1)

    # Standard pendulum equation: I * alpha = Sum of Torques
    # Torque due to gravity: -m*g*L*sin(theta) (assuming theta=0 is vertically down)
    # The JSON data likely has theta=0 as horizontal, so we use sin(theta) if gravity is in Y,
    # or cos(theta) if gravity is in Z. Let's stick with the provided formula.
    # IMPORTANT: Your JSON has gravity in Z = -9.81. If theta is from the vertical, use sin.
    # If theta is from the horizontal, use cos. The code uses sin, assuming theta from vertical.
    torque_gravity = -mass_sq * g * length_com_sq * torch.sin(theta)

    torque_damping = -b_estimated * omega
    
    net_torque = applied_torque_sq + torque_gravity + torque_damping
    
    alpha = net_torque / inertia_yy_sq

    # Semi-implicit Euler integration
    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt 

    return torch.stack([theta_next, omega_next], dim=1)

def train_step(model, optimizer, data):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Get damping prediction from the model
    damping = model(data)
    
    # --- CORRECTED: Call simulate_step_physical directly ---
    # This removes the need for a custom `compute_next_state` method in the model
    next_state = simulate_step_physical(
        x=data.x,
        applied_torque=data.true_torque_t,
        estimated_b=damping,
        dt=data.dt_step,
        mass=data.mass,
        length_com_for_gravity=data.length_com_for_gravity,
        inertia_yy=data.inertia_yy,
        gravity_accel=data.gravity_accel
    )
    
    # Calculate loss against the true next state from the dataset
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
        
        # --- CORRECTED: Call simulate_step_physical directly ---
        next_state = simulate_step_physical(
            x=data.x,
            applied_torque=data.true_torque_t,
            estimated_b=damping,
            dt=data.dt_step,
            mass=data.mass,
            length_com_for_gravity=data.length_com_for_gravity,
            inertia_yy=data.inertia_yy,
            gravity_accel=data.gravity_accel
        )
        
        # Calculate loss
        loss = F.mse_loss(next_state, data.x_next)
    
    return loss.item()

def run_training(model, train_loader, test_loader, device, epochs=200, lr=1e-2, weight_decay=1e-5):
    """Run training with improved learning rate and monitoring"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses, test_losses = [], []
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_total = 0.0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            # The number of graphs in a batch is data.num_graphs
            # The number of nodes is data.num_nodes
            loss = train_step(model, optimizer, data)
            train_loss_total += loss * data.num_graphs
            train_count += data.num_graphs
        
        # Average training loss per graph
        avg_train_loss = train_loss_total / train_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        test_loss_total = 0.0
        test_count = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                loss = validate_step(model, data)
                test_loss_total += loss * data.num_graphs
                test_count += data.num_graphs
        
        # Average test loss per graph
        avg_test_loss = test_loss_total / test_count
        test_losses.append(avg_test_loss)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print metrics
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    print("Training done.")
    return train_losses, test_losses