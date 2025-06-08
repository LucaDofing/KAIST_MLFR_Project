# main.py
import torch
from torch_geometric.loader import DataLoader
from src.datasets import MuJoCoPendulumDataset # Changed from FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training, simulate_step, simulate_step_physical # simulate_step might not be needed here if used only in train
from src.config import (
    TOTAL_SAMPLES_FROM_JSON, TRAIN_SPLIT_RATIO, BATCH_SIZE,
    HIDDEN_DIM, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY # Added HIDDEN_DIM
)
import os # For path joining

def main():
    # Step 1: Load dataset (unsupervised mode)
    # Define the root directory for MuJoCo data
    mujoco_data_dir = os.path.join('data', 'mujoco')
    
    # Ensure the processed directory exists for the dataset
    processed_dir = os.path.join(mujoco_data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # You need to ensure your JSON files are in mujoco_data_dir or mujoco_data_dir/raw
    # For this example, assuming they are directly in mujoco_data_dir
    # If they are in data/mujoco/raw, MuJoCoPendulumDataset should find them
    
    # Make sure raw files are "discoverable" by the dataset class
    # E.g., copy them to data/mujoco/raw if they are not already there.
    # For simplicity, I'll assume they are in data/mujoco for now and glob will pick them up
    # Or, ensure the `raw_file_names` and `process` methods correctly locate them.
    # The MuJoCoPendulumDataset is set up to look in self.root first for the pattern.

    print(f"Looking for JSON files in: {mujoco_data_dir}")
    full_dataset = MuJoCoPendulumDataset(root_dir=mujoco_data_dir, json_files_pattern="*.json", mode="unsupervised")
    print(f"Total samples loaded: {len(full_dataset)}")

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
    model = DampingGCN(hidden_dim=HIDDEN_DIM).to(device) # Use HIDDEN_DIM from config

    # Step 3: Train
    run_training(model, train_loader, test_loader, device, 
                 epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Optional: Run a test prediction on a sample from the dataset
    if len(full_dataset) > 0:
        sample_idx = 0
        sample = full_dataset[sample_idx].to(device)

        with torch.no_grad():
            model.eval()
            pred_damping = model(sample)
            estimated_b_coeff = pred_damping.squeeze()  # <--- hier hinzugefügt

            sample_dt = sample.dt_step.item()
            omega_mean = sample.omega_mean.item()
            omega_std = sample.omega_std.item()

            pred_next = simulate_step_physical(
                x=sample.x,
                applied_torque=sample.true_torque_t,
                estimated_b=estimated_b_coeff,
                dt=sample_dt,
                mass=sample.true_mass,
                length_com_for_gravity=sample.true_length,
                inertia_yy=sample.inertia_yy,
                gravity_accel=sample.gravity_accel,
                omega_mean=omega_mean,
                omega_std=omega_std
            )

        print("\nSample prediction (unsupervised):")
        print("Current state (sin(θ), cos(θ), ω_norm):")
        print(sample.x.cpu().numpy())
        print("Predicted next state (sin(θ), cos(θ), ω_norm):")
        print(pred_next.cpu().numpy())
        print("True next state (sin(θ), cos(θ), ω_norm) (from MuJoCo):")
        print(sample.x_next.cpu().numpy())
        print("Estimated damping per joint (from GNN):")
        print(pred_damping.squeeze().cpu().numpy())
        if hasattr(sample, 'y_true_damping'):
            print("True physical damping per joint (from JSON):")
            print(sample.y_true_damping.cpu().numpy())
        print("Estimated physical damping coeff 'b' (from GNN):")
        print(estimated_b_coeff.cpu().numpy())
        if hasattr(sample, 'y_true_damping_b'):
            print("True physical damping coeff 'b' (from JSON):")
            print(sample.y_true_damping_b.cpu().numpy())
        print(f"  (using mass: {sample.mass.item():.4f}, L_com_gravity: {sample.length_com_for_gravity.item():.4f}, I_yy: {sample.inertia_yy.item():.6e}, dt: {sample.dt_step.item()}, g: {sample.gravity_accel.item()}, omega_mean: {omega_mean:.4f}, omega_std: {omega_std:.4f})")

if __name__ == "__main__":
    main()