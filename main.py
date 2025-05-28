# main.py
import torch
from torch_geometric.loader import DataLoader
from src.datasets import MuJoCoPendulumDataset # Changed from FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training, simulate_step # simulate_step might not be needed here if used only in train
from src.config import (
    TOTAL_SAMPLES_FROM_JSON, TRAIN_SPLIT_RATIO, BATCH_SIZE,
    HIDDEN_DIM, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_CHECKPOINT_DIR # Added HIDDEN_DIM
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
    
    # ... (after run_training call)
    print("Training done.")

    # --- Save the trained model ---
    model_save_path = os.path.join(MODEL_CHECKPOINT_DIR, "gnn_effective_damping_model.pth")
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    # -----------------------------
    # ... after
# train_losses, test_losses = run_training(...)

# Save losses
loss_history_path = os.path.join(MODEL_CHECKPOINT_DIR, f"loss_history_{json_pattern.replace('*','all')}.npz")
np.savez(loss_history_path, train_losses=np.array(train_losses), test_losses=np.array(test_losses))
print(f"Loss history saved to {loss_history_path}")

# ... then call plotting.plot_losses directly here as before for immediate viewing
plot_losses(train_losses, test_losses, save_path=os.path.join(PLOT_DIR, f"loss_curve_simplified_{json_pattern.replace('*','all')}.png"))
    # ... (rest of your sample prediction code)

    # Optional: Run a test prediction on a sample from the dataset
    if len(full_dataset) > 0:
        sample_idx = 0 # Or any other index
        sample = full_dataset[sample_idx].to(device)
        
        with torch.no_grad():
            model.eval() # Ensure model is in eval mode
            pred_damping = model(sample)
            
            # Get dt from the sample itself
            sample_dt = sample.dt_step.item() 
            
            # Use the simulate_step from train.py for consistency
            pred_next = simulate_step(sample.x, pred_damping, dt=sample_dt)

        print("\nSample prediction (unsupervised):")
        # print(f"File source: {full_dataset.dataset.data_list[sample_idx].file_origin if hasattr(full_dataset.dataset, 'data_list') and hasattr(full_dataset.dataset.data_list[sample_idx], 'file_origin') else 'N/A'}") # If you add file_origin to Data
        print("Current state θ, ω:")
        print(sample.x.cpu().numpy())
        print("Predicted next state θ_next, ω_next:")
        print(pred_next.cpu().numpy())
        print("True next state θ_next, ω_next (from MuJoCo):")
        print(sample.x_next.cpu().numpy())
        print("Estimated damping per joint (from GNN):")
        print(pred_damping.squeeze().cpu().numpy())
        if hasattr(sample, 'y_true_damping'):
            print("True physical damping per joint (from JSON):")
            print(sample.y_true_damping.cpu().numpy())
    else:
        print("Dataset is empty, cannot run sample prediction.")

if __name__ == "__main__":
    main()