# analyze_results.py
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from src.datasets import MuJoCoPendulumDataset # Or the dataset class you used
from src.models import DampingGCN
from src.train import simulate_step # The simplified simulate_step from your training
from src.config import (
    BATCH_SIZE, HIDDEN_DIM, PLOT_DIR, MODEL_CHECKPOINT_DIR, TRAIN_SPLIT_RATIO
)
from src.plotting import plot_trajectory_comparison # We can reuse this from src.plotting

def calculate_metrics(model, loader, device, simulate_step_fn):
    model.eval()
    total_mse_loss = 0.0
    total_mae_theta = 0.0
    total_mae_omega = 0.0
    node_count = 0

    all_pred_params = [] # To store predicted effective damping

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            estimated_param = model(data) # GNN predicts effective damping
            all_pred_params.extend(estimated_param.squeeze().cpu().tolist())

            dt = data.dt_step[0].item()
            
            pred_next = simulate_step_fn(data.x, estimated_param, dt)
            
            # MSE for (theta_next, omega_next)
            mse_loss = torch.nn.functional.mse_loss(pred_next, data.x_next, reduction='sum')
            total_mse_loss += mse_loss.item()
            
            # MAE for theta_next
            mae_theta = torch.nn.functional.l1_loss(pred_next[:, 0], data.x_next[:, 0], reduction='sum')
            total_mae_theta += mae_theta.item()
            
            # MAE for omega_next
            mae_omega = torch.nn.functional.l1_loss(pred_next[:, 1], data.x_next[:, 1], reduction='sum')
            total_mae_omega += mae_omega.item()
            
            node_count += data.num_nodes
            
    avg_mse = total_mse_loss / node_count if node_count > 0 else 0
    avg_mae_theta = total_mae_theta / node_count if node_count > 0 else 0
    avg_mae_omega = total_mae_omega / node_count if node_count > 0 else 0
    
    return avg_mse, avg_mae_theta, avg_mae_omega, all_pred_params

def main_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    model_path = os.path.join(MODEL_CHECKPOINT_DIR, "gnn_effective_damping_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train and save the model first using main.py.")
        return
        
    model = DampingGCN(hidden_dim=HIDDEN_DIM).to(device) # Ensure params match saved model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # --- Load Dataset ---
    # Use the same dataset configuration as when the model was trained
    data_root_dir = os.path.join('data', 'mujoco')
    json_pattern = "1_link_no_torque.json" # Or "*.json" if that's what you used for the 0.0 MSE run

    # Important: Ensure this dataset instance uses the SAME processed file
    # that was generated during the training run you want to analyze.
    # If MuJoCoPendulumDataset has a fixed processed_file_names[0], it should be fine.
    print(f"Loading dataset from '{data_root_dir}' matching '{json_pattern}'")
    full_dataset = MuJoCoPendulumDataset(
        root_dir=data_root_dir,
        json_files_pattern=json_pattern
    )
    
    if len(full_dataset) < 2:
        print("Dataset too small for analysis.")
        return

    actual_total_samples = len(full_dataset)
    num_train = int(TRAIN_SPLIT_RATIO * actual_total_samples)
    if num_train == 0 and actual_total_samples > 1 : num_train = 1 
    if num_train == actual_total_samples and actual_total_samples > 1 : num_train = actual_total_samples - 1
    num_test = actual_total_samples - num_train

    # We typically evaluate on the test set
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_test], generator=torch.Generator().manual_seed(42) # Use same seed
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")


    # --- Calculate Metrics ---
    # Use the SAME simulate_step function that was used in training
    # This is the simplified one for your "effective damping" model
    current_simulate_step_fn = simulate_step

    avg_mse, avg_mae_theta, avg_mae_omega, predicted_params = calculate_metrics(model, test_loader, device, current_simulate_step_fn)
    print(f"\n--- Metrics on Test Set ---")
    print(f"Average MSE (next state prediction): {avg_mse:.8e}")
    print(f"Average MAE Theta_next: {avg_mae_theta:.8e}")
    print(f"Average MAE Omega_next: {avg_mae_omega:.8e}")

    if predicted_params:
        predicted_params_np = np.array(predicted_params)
        print(f"Statistics for GNN's Estimated Effective Damping on Test Set:")
        print(f"  Min: {predicted_params_np.min():.4f}")
        print(f"  Max: {predicted_params_np.max():.4f}")
        print(f"  Mean: {predicted_params_np.mean():.4f}")
        print(f"  Std: {predicted_params_np.std():.4f}")

        # Histogram of predicted effective damping values
        plt.figure(figsize=(8, 6))
        plt.hist(predicted_params_np, bins=50, edgecolor='black')
        plt.title("Distribution of Estimated Effective Damping (Test Set)")
        plt.xlabel("Effective Damping Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        hist_save_path = os.path.join(PLOT_DIR, f"effective_damping_dist_{json_pattern.replace('*','all')}.png")
        os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
        plt.savefig(hist_save_path)
        print(f"Effective damping distribution plot saved to {hist_save_path}")
        plt.close()


    # --- Plot Trajectories ---
    # Use the full_dataset for plotting to get potentially longer/contiguous sequences
    # Pass the correct simulate_step_fn
    os.makedirs(PLOT_DIR, exist_ok=True) # Ensure plot directory exists
    plot_trajectory_comparison(model, full_dataset, device,
                               simulate_step_fn=current_simulate_step_fn,
                               num_trajectories=min(3, len(full_dataset) // 100 if len(full_dataset) >=100 else 1),
                               steps_to_plot=200, # Plot more steps
                               save_dir=PLOT_DIR,
                               use_physical_model=False) # Explicitly state we are using the simplified model context

    print("\nAnalysis complete. Plots saved in:", PLOT_DIR)

if __name__ == "__main__":
    # You might need to adjust PYTHONPATH if src is not found directly
    # e.g., export PYTHONPATH=$PYTHONPATH:$(pwd) before running
    main_analysis()