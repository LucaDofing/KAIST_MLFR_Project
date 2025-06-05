import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
from src.config import BATCH_SIZE, TRAIN_SPLIT_RATIO, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.models import DampingGCN
from src.datasets import MuJoCoPendulumDataset
from src.train import run_training
from src.preprocess import AngleNormalizer

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
    
    # Calculate train/test split
    n_samples = len(dataset)
    n_train = int(n_samples * TRAIN_SPLIT_RATIO)
    n_test = n_samples - n_train
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    print(f"Total samples: {n_samples}")
    print(f"Training samples: {n_train}")
    print(f"Testing samples: {n_test}")
    
    # Create angle normalizer and fit on training data
    normalizer = AngleNormalizer()
    normalizer.fit(train_dataset)
    
    # Save normalizer parameters for future use
    os.makedirs('models', exist_ok=True)
    normalizer.save('models/angle_normalizer.pt')
    
    # Transform datasets
    train_dataset_transformed = normalizer.transform_dataset(train_dataset)
    test_dataset_transformed = normalizer.transform_dataset(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset_transformed, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset_transformed, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = DampingGCN().to(device)
    
    # Train model with combined supervised and unsupervised learning
    # Try different lambda values for the supervised component
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for lambda_supervised in lambda_values:
        print(f"\n=== Training with lambda_supervised = {lambda_supervised} ===")
        
        # Reinitialize model for each run
        model = DampingGCN().to(device)
        
        # Train model
        run_training(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            normalizer=normalizer,
            epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lambda_supervised=lambda_supervised
        )
        
        # Save trained model
        model_path = f'models/damping_gcn_model_lambda_{lambda_supervised}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save the final model (with lambda=0.5) as the default model
    torch.save(model.state_dict(), 'models/damping_gcn_model.pt')
    print("Final model saved to models/damping_gcn_model.pt")

if __name__ == "__main__":
    main() 