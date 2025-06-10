# main.py
import torch
from torch_geometric.loader import DataLoader
from src.datasets import MuJoCoPendulumDataset
from src.models import DampingGCN
from src.train import run_training, simulate_step_physical
from src.config import (
    TRAIN_SPLIT_RATIO, BATCH_SIZE,
    HIDDEN_DIM, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
)
import os
import numpy as np
import sys

# --- START: ADDED CODE TO HANDLE IMPORTS AND PLOTTING ---

# Add the '2_GNN' directory to the Python path to find the 'plotting' module
# This makes the script runnable from the project root directory
gnn_dir = os.path.dirname(os.path.abspath(__file__))
if gnn_dir not in sys.path:
    sys.path.append(gnn_dir)

from plotting.plot_loss import plot_loss_curve
from plotting.plot_damping_dist import plot_damping_distribution
from plotting.plot_prediction_scatter import plot_prediction_scatter

# --- END: ADDED CODE ---

def main():
    # Step 1: Load dataset
    mujoco_data_dir = os.path.join('data', 'mujoco')
    processed_dir = os.path.join(mujoco_data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Looking for JSON files in: {mujoco_data_dir}")
    full_dataset = MuJoCoPendulumDataset(root_dir=mujoco_data_dir, json_files_pattern="*.json", mode="unsupervised")
    print(f"Total samples loaded: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    num_train = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    num_test = len(full_dataset) - num_train
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_test], generator=torch.Generator().manual_seed(42)
    )
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = DampingGCN(hidden_dim=HIDDEN_DIM).to(device)

    # Step 3: Train and capture the loss history
    train_losses, test_losses = run_training(
        model, train_loader, test_loader, device, 
        epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    # --- START: REVISED EVALUATION AND PLOTTING SECTION ---

    print("\n--- Starting Final Evaluation and Plotting ---")

    # Step 4: Evaluate on test set (one combined loop for efficiency)
    model.eval()
    estimated_b_values = []
    true_b_values = []
    all_pred_next = []
    all_true_next = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get damping prediction for histogram
            damping = model(data)
            estimated_b_values.extend(damping.squeeze().cpu().numpy())
            true_b_values.extend(data.y_true_damping.squeeze().cpu().numpy())
            
            # Compute next state using full physics simulation for scatter plot
            pred_next = simulate_step_physical(
                x=data.x,
                applied_torque=data.true_torque_t,
                estimated_b=damping,
                dt=data.dt_step,
                mass=data.mass,
                length_com_for_gravity=data.length_com_for_gravity,
                inertia_yy=data.inertia_yy,
                gravity_accel=data.gravity_accel
            )
            all_pred_next.append(pred_next.cpu())
            all_true_next.append(data.x_next.cpu())

    # Convert lists to numpy arrays for plotting
    estimated_b_values = np.array(estimated_b_values)
    true_b_value = true_b_values[0] if true_b_values else 0.0
    
    all_pred_next_np = torch.cat(all_pred_next, dim=0).numpy()
    all_true_next_np = torch.cat(all_true_next, dim=0).numpy()

    # Step 5: Generate and Save Plots
    print("\nGenerating plots...")
    plot_loss_curve(train_losses, test_losses)
    plot_damping_distribution(estimated_b_values, true_b_value)
    plot_prediction_scatter(all_pred_next_np, all_true_next_np)
    
    print("\n--- All plots have been generated and saved in the 'results' directory. ---")

    # --- END: REVISED EVALUATION AND PLOTTING SECTION ---

    # Optional: Run a test prediction on a single sample for console output
    if len(full_dataset) > 0:
        sample_idx = 0
        sample = full_dataset[sample_idx].to(device)

        with torch.no_grad():
            model.eval()
            pred_damping_tensor = model(sample)
            
            pred_next_sample = simulate_step_physical(
                x=sample.x,
                applied_torque=sample.true_torque_t,
                estimated_b=pred_damping_tensor,
                dt=sample.dt_step,
                mass=sample.true_mass,
                length_com_for_gravity=sample.true_length,
                inertia_yy=sample.inertia_yy,
                gravity_accel=sample.gravity_accel
            )

        print("\n--- Single Sample Prediction (for debugging) ---")
        print("Current state θ, ω:")
        print(sample.x.cpu().numpy())
        print("Predicted next state θ_next, ω_next:")
        print(pred_next_sample.cpu().numpy())
        print("True next state θ_next, ω_next (from MuJoCo):")
        print(sample.x_next.cpu().numpy())
        if hasattr(sample, 'y_true_damping'):
            print("True physical damping per joint (from JSON):")
            print(sample.y_true_damping.cpu().numpy())
        print("Estimated physical damping coeff 'b' (from GNN):")
        print(pred_damping_tensor.squeeze().cpu().numpy())
        print(f"  (using mass: {sample.mass.item():.4f}, L_com_gravity: {sample.length_com_for_gravity.item():.4f}, I_yy: {sample.inertia_yy.item():.6e}, dt: {sample.dt_step.item()}, g: {sample.gravity_accel.item()})")

if __name__ == "__main__":
    main()