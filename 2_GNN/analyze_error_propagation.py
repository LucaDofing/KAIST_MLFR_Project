# analyze_error_propagation.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# --- We need to import or copy the necessary components from your project ---
from src.datasets import MuJoCoPendulumDataset
from src.models import DampingGCN
from src.config import HIDDEN_DIM

# Copy the simulation function here for a self-contained script
def simulate_step_physical(x, applied_torque, estimated_b, dt, mass, length_com_for_gravity, inertia_yy, gravity_accel):
    """ Differentiable physics simulation step. (Copied from src/train.py) """
    theta, omega = x[:, 0], x[:, 1]
    
    # Ensure all tensors are on the same device and have compatible shapes
    device = x.device
    b_estimated = estimated_b.squeeze(-1).to(device)
    mass_sq = mass.squeeze(-1).to(device)
    length_com_sq = length_com_for_gravity.squeeze(-1).to(device)
    inertia_yy_sq = inertia_yy.squeeze(-1).to(device)
    applied_torque_sq = applied_torque.squeeze(-1).to(device)
    g = gravity_accel.squeeze(-1).to(device)
    dt = dt.to(device)

    # Torque due to gravity (assuming theta is from vertical)
    torque_gravity = -mass_sq * g * length_com_sq * torch.sin(theta)
    torque_damping = -b_estimated * omega
    
    net_torque = applied_torque_sq + torque_gravity + torque_damping
    alpha = net_torque / inertia_yy_sq

    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt 

    return torch.stack([theta_next, omega_next], dim=1)

def analyze_sample_loss_landscape(model, sample_data, device):
    """
    Analyzes the prediction error for a single data sample across a range of
    damping coefficients.
    """
    model.eval()
    sample_data = sample_data.to(device)

    # --- Step 1: Get the key values for this sample ---
    with torch.no_grad():
        b_gnn_pred = model(sample_data).squeeze().item()

    b_true = sample_data.y_true_damping.item()
    x_next_true = sample_data.x_next
    
    print(f"--- Analyzing Sample ---")
    print(f"True Damping (b_true): {b_true:.4f}")
    print(f"GNN Predicted Damping (b_hat): {b_gnn_pred:.4f}")

    # --- Step 2: Sweep through a range of 'b' values and compute loss ---
    b_range = np.linspace(-0.5, 2.0, 300) # A wide range of potential b values
    mse_losses = []

    for b_test in b_range:
        b_tensor = torch.tensor([[b_test]], dtype=torch.float32)
        
        with torch.no_grad():
            x_next_pred = simulate_step_physical(
                x=sample_data.x,
                applied_torque=sample_data.true_torque_t,
                estimated_b=b_tensor,
                dt=sample_data.dt_step,
                mass=sample_data.mass,
                length_com_for_gravity=sample_data.length_com_for_gravity,
                inertia_yy=sample_data.inertia_yy,
                gravity_accel=sample_data.gravity_accel
            )
            loss = F.mse_loss(x_next_pred, x_next_true).item()
            mse_losses.append(loss)

    # --- Step 3: Calculate the specific error points for comparison ---
    # Error if we used the TRUE damping in our simple simulator
    b_true_tensor = torch.tensor([[b_true]], dtype=torch.float32)
    x_next_at_b_true = simulate_step_physical(x=sample_data.x, applied_torque=sample_data.true_torque_t, estimated_b=b_true_tensor, dt=sample_data.dt_step, mass=sample_data.mass, length_com_for_gravity=sample_data.length_com_for_gravity, inertia_yy=sample_data.inertia_yy, gravity_accel=sample_data.gravity_accel)
    mse_at_b_true = F.mse_loss(x_next_at_b_true, x_next_true).item()

    # Error if we used the GNN's predicted damping
    b_gnn_tensor = torch.tensor([[b_gnn_pred]], dtype=torch.float32)
    x_next_at_b_gnn = simulate_step_physical(x=sample_data.x, applied_torque=sample_data.true_torque_t, estimated_b=b_gnn_tensor, dt=sample_data.dt_step, mass=sample_data.mass, length_com_for_gravity=sample_data.length_com_for_gravity, inertia_yy=sample_data.inertia_yy, gravity_accel=sample_data.gravity_accel)
    mse_at_b_gnn = F.mse_loss(x_next_at_b_gnn, x_next_true).item()

    print(f"\nPrediction MSE using TRUE damping (b={b_true:.2f}):   {mse_at_b_true:.6f}")
    print(f"Prediction MSE using GNN damping (b={b_gnn_pred:.2f}):  {mse_at_b_gnn:.6f}")
    
    # --- Step 4: Plot the loss landscape ---
    plt.figure(figsize=(12, 7))
    plt.plot(b_range, mse_losses, label='Prediction Error (MSE) Landscape', linewidth=2)
    
    # Mark the true damping value
    plt.axvline(x=b_true, color='r', linestyle='--', label=f'True Damping b = {b_true:.2f}\n(MSE = {mse_at_b_true:.4f})')
    plt.scatter(b_true, mse_at_b_true, s=100, c='r', zorder=5)

    # Mark the GNN's prediction
    plt.axvline(x=b_gnn_pred, color='g', linestyle='--', label=f'GNN Predicted Damping b̂ = {b_gnn_pred:.2f}\n(MSE = {mse_at_b_gnn:.4f})')
    plt.scatter(b_gnn_pred, mse_at_b_gnn, s=100, c='g', zorder=5)

    # Mark the minimum of the curve
    min_loss_idx = np.argmin(mse_losses)
    min_loss_b = b_range[min_loss_idx]
    plt.axvline(x=min_loss_b, color='purple', linestyle=':', label=f'Optimal b for this simulator = {min_loss_b:.2f}')


    plt.title('Analysis of Prediction Error vs. Damping Coefficient')
    plt.xlabel('Damping Coefficient "b" used in Simulation')
    plt.ylabel('Next-State Prediction MSE')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    # Save the plot
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, 'error_propagation_landscape.png'))
    print(f"\nSaved loss landscape plot to '{results_dir}/error_propagation_landscape.png'")
    plt.show()


if __name__ == '__main__':
    # --- Step 1: Load the trained model and dataset ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make sure you have a trained model saved as 'damping_gcn.pth'
    model_path = 'damping_gcn.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run train.py first to generate it.")

    model = DampingGCN(hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from '{model_path}'")

    # Load the dataset
    mujoco_data_dir = os.path.join('data', 'mujoco')
    full_dataset = MuJoCoPendulumDataset(root_dir=mujoco_data_dir, json_files_pattern="*.json")
    
    # --- Step 2: Select a sample for analysis ---
    # Pick a sample that is interesting. For example, one where the GNN's prediction
    # was far from the true value. Or just pick one randomly.
    sample_idx = 150 # You can change this index
    sample_data = full_dataset[sample_idx]

    # --- Step 3: Run the analysis ---
    analyze_sample_loss_landscape(model, sample_data, device)