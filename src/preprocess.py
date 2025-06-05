import torch
import numpy as np
from torch_geometric.data import Data

class AngleNormalizer:
    """
    Transforms theta angles to sin(theta) and cos(theta) for better representation
    and normalizes omega values using mean and std from training data.
    """
    def __init__(self):
        self.omega_mean = None
        self.omega_std = None
        self.is_fitted = False
    
    def fit(self, dataset):
        """Calculate mean and std of omega from the training dataset"""
        all_omega = []
        for data in dataset:
            # Extract omega (second feature)
            omega = data.x[:, 1]
            all_omega.append(omega)
        
        all_omega = torch.cat(all_omega)
        self.omega_mean = all_omega.mean().item()
        self.omega_std = all_omega.std().item()
        self.is_fitted = True
        
        print(f"Fitted omega normalizer: mean={self.omega_mean:.4f}, std={self.omega_std:.4f}")
        return self
    
    def transform_features(self, data):
        """
        Transform a Data object by:
        1. Converting theta to sin(theta) and cos(theta)
        2. Normalizing omega
        """
        if not self.is_fitted:
            raise ValueError("AngleNormalizer must be fitted before transform")
        
        # Extract features
        theta = data.x[:, 0]
        omega = data.x[:, 1]
        
        # Transform theta to sin and cos
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Normalize omega
        omega_norm = (omega - self.omega_mean) / self.omega_std
        
        # Create new feature matrix [sin(theta), cos(theta), normalized_omega]
        new_x = torch.stack([sin_theta, cos_theta, omega_norm], dim=1)
        
        # Create a new Data object with transformed features
        transformed_data = Data(
            x=new_x,
            edge_index=data.edge_index,
            y=data.y if hasattr(data, 'y') else None
        )
        
        # Copy all other attributes
        for key, value in data:
            if key not in ['x', 'edge_index', 'y']:
                transformed_data[key] = value
                
        # If x_next exists (for unsupervised learning), transform it too
        if hasattr(data, 'x_next'):
            theta_next = data.x_next[:, 0]
            omega_next = data.x_next[:, 1]
            
            sin_theta_next = torch.sin(theta_next)
            cos_theta_next = torch.cos(theta_next)
            omega_next_norm = (omega_next - self.omega_mean) / self.omega_std
            
            new_x_next = torch.stack([sin_theta_next, cos_theta_next, omega_next_norm], dim=1)
            transformed_data.x_next = new_x_next
            
        return transformed_data
    
    def inverse_transform_omega(self, omega_norm):
        """Convert normalized omega back to original scale"""
        return omega_norm * self.omega_std + self.omega_mean
    
    def transform_dataset(self, dataset):
        """Transform an entire dataset"""
        transformed_data_list = []
        for data in dataset:
            transformed_data = self.transform_features(data)
            transformed_data_list.append(transformed_data)
        return transformed_data_list
    
    def save(self, path):
        """Save normalizer parameters"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
        torch.save({
            'omega_mean': self.omega_mean,
            'omega_std': self.omega_std
        }, path)
        
    def load(self, path):
        """Load normalizer parameters"""
        params = torch.load(path)
        self.omega_mean = params['omega_mean']
        self.omega_std = params['omega_std']
        self.is_fitted = True
        return self 