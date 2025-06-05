import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from src.config import BATCH_SIZE
from src.models import DampingGCN
from src.datasets import MuJoCoPendulumDataset
from src.preprocess import AngleNormalizer
from src.train import simulate_step

def evaluate_model(model, test_loader, device, normalizer):
    model.eval()
    all_true_dampings = []
    all_pred_dampings = []
    all_errors = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get model predictions
            estimated_damping = model(data)
            
            # Store true and predicted dampings
            true_damping = data.y_true_damping.cpu().numpy()
            pred_damping = estimated_damping.cpu().numpy()
            
            all_true_dampings.append(true_damping)
            all_pred_dampings.append(pred_damping)
            
            # Calculate error
            error = np.abs(true_damping - pred_damping)
            all_errors.append(error)
    
    # Concatenate results
    all_true_dampings = np.concatenate(all_true_dampings)
    all_pred_dampings = np.concatenate(all_pred_dampings)
    all_errors = np.concatenate(all_errors)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_true_dampings - all_pred_dampings))
    mse = np.mean(np.square(all_true_dampings - all_pred_dampings))
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_true_dampings, all_pred_dampings, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('True Damping')
    plt.ylabel('Predicted Damping')
    plt.title('True vs Predicted Damping Coefficients')
    plt.grid(True)
    
    return mae, mse, rmse, all_true_dampings, all_pred_dampings

def simulate_trajectory(model, initial_state, dt, steps, normalizer, device):
    """
    Simulate a trajectory using the trained model
    
    Args:
        model: Trained GNN model
        initial_state: Initial state tensor [theta, omega]
        dt: Time step
        steps: Number of steps to simulate
        normalizer: AngleNormalizer instance
        device: Device to run on
    
    Returns:
        Simulated trajectory
    """
    model.eval()
    
    # Transform initial state
    theta, omega = initial_state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    norm_omega = (omega - normalizer.omega_mean) / normalizer.omega_std
    
    # Create feature tensor
    x = torch.tensor([[sin_theta, cos_theta, norm_omega]], dtype=torch.float32).to(device)
    
    # Create dummy graph data
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    
    # Lists to store trajectory
    thetas = [theta]
    omegas = [omega]
    dampings = []
    
    # Simulate trajectory
    with torch.no_grad():
        for _ in range(steps):
            # Create dummy data object
            class DummyData:
                pass
            
            data = DummyData()
            data.x = x
            data.edge_index = edge_index
            
            # Get damping prediction
            estimated_damping = model(data)
            dampings.append(estimated_damping.item())
            
            # Simulate one step
            x_next = simulate_step(x, estimated_damping, dt, normalizer)
            
            # Extract theta and omega from transformed features
            sin_theta_next = x_next[0, 0].item()
            cos_theta_next = x_next[0, 1].item()
            norm_omega_next = x_next[0, 2].item()
            
            # Convert back to original representation
            theta_next = np.arctan2(sin_theta_next, cos_theta_next)
            omega_next = normalizer.inverse_transform_omega(norm_omega_next)
            
            # Store results
            thetas.append(theta_next)
            omegas.append(omega_next)
            
            # Update state
            x = x_next
    
    return thetas, omegas, dampings

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = MuJoCoPendulumDataset(
        root_dir='data/mujoco',
        json_files_pattern="*.json", 
        mode="unsupervised"
    )
    
    # Load normalizer
    normalizer = AngleNormalizer()
    normalizer.load('models/angle_normalizer.pt')
    
    # Transform dataset
    test_dataset_transformed = normalizer.transform_dataset(dataset)
    test_loader = DataLoader(test_dataset_transformed, batch_size=BATCH_SIZE)
    
    # Evaluate models with different lambda values
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, lambda_val in enumerate(lambda_values):
        model_path = f'models/damping_gcn_model_lambda_{lambda_val}.pt'
        
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping.")
            continue
        
        print(f"\n=== Evaluating model with lambda = {lambda_val} ===")
        
        # Load model
        model = DampingGCN().to(device)
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate model
        mae, mse, rmse, true_dampings, pred_dampings = evaluate_model(model, test_loader, device, normalizer)
        
        # Store results
        results[lambda_val] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        # Plot results for this model
        plt.subplot(2, 3, i+1)
        plt.scatter(true_dampings, pred_dampings, alpha=0.3)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f'Lambda = {lambda_val}')
        plt.xlabel('True Damping')
        plt.ylabel('Predicted Damping')
        plt.grid(True)
        
        # Simulate a trajectory
        initial_state = (0.5, 0.0)  # Initial theta and omega
        dt = 0.01  # Time step
        steps = 200  # Number of steps to simulate
        
        thetas, omegas, dampings = simulate_trajectory(model, initial_state, dt, steps, normalizer, device)
        
        # Plot trajectory for this model
        plt.figure(figsize=(12, 8))
        
        time = np.arange(len(thetas)) * dt
        time_steps = time[:-1]  # Adjust time array to match omegas and dampings
        
        plt.subplot(3, 1, 1)
        plt.plot(time, thetas)
        plt.ylabel('Theta (rad)')
        plt.title(f'Simulated Pendulum Trajectory (Lambda = {lambda_val})')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_steps, omegas[:-1])  # Use time_steps and exclude last omega
        plt.ylabel('Omega (rad/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_steps, dampings)  # Use time_steps
        plt.xlabel('Time (s)')
        plt.ylabel('Predicted Damping')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'trajectory_simulation_lambda_{lambda_val}.png')
        print(f"Trajectory plot saved as trajectory_simulation_lambda_{lambda_val}.png")
    
    # Save the comparison plot
    plt.tight_layout()
    plt.savefig('damping_predictions_comparison.png')
    print("Comparison plot saved as damping_predictions_comparison.png")
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    print("Lambda\tMAE\tMSE\tRMSE")
    for lambda_val in lambda_values:
        if lambda_val in results:
            r = results[lambda_val]
            print(f"{lambda_val:.1f}\t{r['mae']:.4f}\t{r['mse']:.4f}\t{r['rmse']:.4f}")

if __name__ == "__main__":
    import os
    main() 