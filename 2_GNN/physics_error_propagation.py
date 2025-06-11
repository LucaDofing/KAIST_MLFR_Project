import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from src.datasets import MuJoCoPendulumDataset
from src.models import DampingGCN
import os
from src.config import BATCH_SIZE, HIDDEN_DIM

class PhysicsErrorAnalyzer:
    """
    Class to analyze error propagation through the physics calculations
    from GNN output to final damping coefficient
    """
    def __init__(self, model_path='model_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        # Load dataset
        mujoco_data_dir = os.path.join('data', 'mujoco')
        self.dataset = MuJoCoPendulumDataset(
            root_dir=mujoco_data_dir, 
            json_files_pattern="*.json", 
            mode="unsupervised"
        )
        print(f"Total samples loaded: {len(self.dataset)}")
    
    def _load_model(self, model_path):
        """Load a trained GNN model"""
        model = DampingGCN(hidden_dim=HIDDEN_DIM).to(self.device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Using untrained model.")
        
        model.eval()
        return model
    
    def calculate_damping_from_states(self, current_state, next_state, torque, dt, mass, length, inertia, gravity):
        """
        Reverse-calculate the damping coefficient from states
        
        This is the inverse of the physics simulation step:
        θ_next = θ + ω * dt
        ω_next = ω + (τ - b*ω - m*g*L*sin(θ)) / I * dt
        
        Solving for b:
        b = (τ - I*(ω_next - ω)/dt - m*g*L*sin(θ)) / ω
        """
        theta, omega = current_state[0], current_state[1]
        _, omega_next = next_state[0], next_state[1]
        
        # Calculate angular acceleration from states
        alpha = (omega_next - omega) / dt
        
        # Calculate gravity torque
        gravity_torque = mass * gravity * length * np.sin(theta)
        
        # Solve for damping coefficient
        # τ - b*ω - m*g*L*sin(θ) = I*α
        # b*ω = τ - m*g*L*sin(θ) - I*α
        # b = (τ - m*g*L*sin(θ) - I*α) / ω
        
        if abs(omega) < 1e-6:  # Avoid division by zero
            return float('nan')
        
        damping_coeff = (torque - gravity_torque - inertia * alpha) / omega
        return damping_coeff
    
    def analyze_samples(self, num_samples=50):
        """Analyze error propagation through physics calculations"""
        # Select a subset of samples for analysis
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        results = {
            'true_b': [],
            'gnn_b': [],
            'reverse_calculated_b': [],
            'gnn_error': [],
            'physics_error': [],
            'total_error': []
        }
        
        for idx in indices:
            sample = self.dataset[idx].to(self.device)
            
            # Skip samples without ground truth damping
            if not hasattr(sample, 'y_true_damping_b'):
                continue
            
            with torch.no_grad():
                # Get GNN prediction
                pred_damping = self.model(sample)
                estimated_b_coeff = pred_damping.squeeze().cpu().numpy()
            
            # Extract true values and parameters
            true_b_coeff = sample.y_true_damping_b.cpu().numpy()
            current_state = sample.x.cpu().numpy()
            next_state = sample.x_next.cpu().numpy()
            torque = sample.true_torque_t.cpu().numpy()
            dt = sample.dt_step.item()
            mass = sample.true_mass.item()
            length = sample.true_length.item()
            inertia = sample.inertia_yy.item()
            gravity = sample.gravity_accel.item()
            
            # Calculate damping coefficient from states (reverse calculation)
            reverse_b = self.calculate_damping_from_states(
                current_state, next_state, torque, dt, mass, length, inertia, gravity
            )
            
            # Calculate errors
            gnn_error = 100 * abs(estimated_b_coeff - true_b_coeff) / abs(true_b_coeff) if true_b_coeff != 0 else float('inf')
            physics_error = 100 * abs(reverse_b - true_b_coeff) / abs(true_b_coeff) if true_b_coeff != 0 else float('inf')
            total_error = 100 * abs(estimated_b_coeff - reverse_b) / abs(reverse_b) if reverse_b != 0 else float('inf')
            
            # Store results
            results['true_b'].append(true_b_coeff)
            results['gnn_b'].append(estimated_b_coeff)
            results['reverse_calculated_b'].append(reverse_b)
            results['gnn_error'].append(gnn_error)
            results['physics_error'].append(physics_error)
            results['total_error'].append(total_error)
        
        return results
    
    def analyze_error_sensitivity(self, sample_idx=0, error_range=(-50, 50), num_points=20):
        """
        Analyze how errors in GNN output propagate to final state prediction
        by introducing controlled errors to the damping coefficient
        """
        sample = self.dataset[sample_idx].to(self.device)
        
        # Skip samples without ground truth damping
        if not hasattr(sample, 'y_true_damping_b'):
            print("Sample does not have ground truth damping")
            return None
        
        true_b_coeff = sample.y_true_damping_b.cpu().numpy()
        
        # Extract parameters for simulation
        current_state = sample.x.cpu().numpy()
        next_state_true = sample.x_next.cpu().numpy()
        torque = sample.true_torque_t.cpu().numpy()
        dt = sample.dt_step.item()
        mass = sample.true_mass.item()
        length = sample.true_length.item()
        inertia = sample.inertia_yy.item()
        gravity = sample.gravity_accel.item()
        
        # Create array of error percentages to test
        error_percentages = np.linspace(error_range[0], error_range[1], num_points)
        
        results = {
            'error_percentages': error_percentages,
            'damping_values': [],
            'theta_errors': [],
            'omega_errors': []
        }
        
        for error_pct in error_percentages:
            # Apply error to true damping coefficient
            perturbed_b = true_b_coeff * (1 + error_pct/100)
            results['damping_values'].append(perturbed_b)
            
            # Simulate next state with perturbed damping
            next_state_pred = self.simulate_next_state(
                current_state, perturbed_b, torque, dt, mass, length, inertia, gravity
            )
            
            # Calculate state errors
            theta_error = 100 * abs(next_state_pred[0] - next_state_true[0]) / (abs(next_state_true[0]) + 1e-10)
            omega_error = 100 * abs(next_state_pred[1] - next_state_true[1]) / (abs(next_state_true[1]) + 1e-10)
            
            results['theta_errors'].append(theta_error)
            results['omega_errors'].append(omega_error)
        
        return results
    
    def simulate_next_state(self, current_state, damping_coeff, torque, dt, mass, length, inertia, gravity):
        """
        Simulate next state using physics equations
        
        θ_next = θ + ω * dt
        ω_next = ω + (τ - b*ω - m*g*L*sin(θ)) / I * dt
        """
        theta, omega = current_state
        
        # Calculate angular acceleration
        gravity_torque = mass * gravity * length * np.sin(theta)
        damping_torque = damping_coeff * omega
        net_torque = torque - damping_torque - gravity_torque
        alpha = net_torque / inertia
        
        # Update state
        theta_next = theta + omega * dt
        omega_next = omega + alpha * dt
        
        return np.array([theta_next, omega_next])
    
    def plot_error_distributions(self, results):
        """Plot distributions of errors at different stages"""
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Filter out infinities and NaNs
        gnn_errors = np.array(results['gnn_error'])
        gnn_errors = gnn_errors[np.isfinite(gnn_errors)]
        
        physics_errors = np.array(results['physics_error'])
        physics_errors = physics_errors[np.isfinite(physics_errors)]
        
        total_errors = np.array(results['total_error'])
        total_errors = total_errors[np.isfinite(total_errors)]
        
        # Plot GNN error distribution
        axs[0].hist(gnn_errors, bins=20, alpha=0.7)
        axs[0].set_title('GNN Damping Coefficient Error Distribution')
        axs[0].set_xlabel('Percentage Error (%)')
        axs[0].set_ylabel('Frequency')
        axs[0].axvline(np.mean(gnn_errors), color='r', linestyle='dashed', linewidth=1)
        axs[0].text(0.95, 0.95, f'Mean Error: {np.mean(gnn_errors):.2f}%', 
                   transform=axs[0].transAxes, ha='right', va='top')
        
        # Plot physics error distribution
        axs[1].hist(physics_errors, bins=20, alpha=0.7)
        axs[1].set_title('Physics Calculation Error Distribution')
        axs[1].set_xlabel('Percentage Error (%)')
        axs[1].set_ylabel('Frequency')
        axs[1].axvline(np.mean(physics_errors), color='r', linestyle='dashed', linewidth=1)
        axs[1].text(0.95, 0.95, f'Mean Error: {np.mean(physics_errors):.2f}%', 
                   transform=axs[1].transAxes, ha='right', va='top')
        
        # Plot total error distribution
        axs[2].hist(total_errors, bins=20, alpha=0.7)
        axs[2].set_title('Total Error Distribution (GNN vs Reverse Calculation)')
        axs[2].set_xlabel('Percentage Error (%)')
        axs[2].set_ylabel('Frequency')
        axs[2].axvline(np.mean(total_errors), color='r', linestyle='dashed', linewidth=1)
        axs[2].text(0.95, 0.95, f'Mean Error: {np.mean(total_errors):.2f}%', 
                   transform=axs[2].transAxes, ha='right', va='top')
        
        plt.tight_layout()
        plt.savefig('physics_error_distribution.png')
        plt.show()
    
    def plot_true_vs_predicted(self, results):
        """Plot true vs predicted values for damping coefficient"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Filter out infinities and NaNs
        mask = np.isfinite(results['reverse_calculated_b'])
        true_b = np.array(results['true_b'])[mask]
        gnn_b = np.array(results['gnn_b'])[mask]
        reverse_b = np.array(results['reverse_calculated_b'])[mask]
        
        # Plot true vs GNN predicted
        axs[0].scatter(true_b, gnn_b, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(true_b.min(), gnn_b.min())
        max_val = max(true_b.max(), gnn_b.max())
        axs[0].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axs[0].set_title('True vs GNN Predicted Damping Coefficient')
        axs[0].set_xlabel('True b Coefficient')
        axs[0].set_ylabel('GNN Predicted b Coefficient')
        
        # Plot true vs reverse calculated
        axs[1].scatter(true_b, reverse_b, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(true_b.min(), reverse_b.min())
        max_val = max(true_b.max(), reverse_b.max())
        axs[1].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axs[1].set_title('True vs Reverse Calculated b Coefficient')
        axs[1].set_xlabel('True b Coefficient')
        axs[1].set_ylabel('Reverse Calculated b Coefficient')
        
        plt.tight_layout()
        plt.savefig('physics_true_vs_predicted.png')
        plt.show()
    
    def plot_error_sensitivity(self, sensitivity_results):
        """Plot how errors in damping coefficient affect state prediction"""
        if sensitivity_results is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot how errors in damping affect state prediction errors
        ax.plot(sensitivity_results['error_percentages'], 
                sensitivity_results['theta_errors'], 
                'b-', label='θ (angle) error')
        ax.plot(sensitivity_results['error_percentages'], 
                sensitivity_results['omega_errors'], 
                'r-', label='ω (angular velocity) error')
        
        ax.set_title('Sensitivity of State Prediction to Damping Coefficient Errors')
        ax.set_xlabel('Error in Damping Coefficient (%)')
        ax.set_ylabel('Resulting Error in State Prediction (%)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('error_sensitivity.png')
        plt.show()

def main():
    # Create analyzer
    analyzer = PhysicsErrorAnalyzer()
    
    # Analyze error propagation
    print("Analyzing error propagation through physics calculations...")
    results = analyzer.analyze_samples(num_samples=100)
    
    # Print summary statistics
    print("\n===== ERROR PROPAGATION ANALYSIS =====")
    
    gnn_errors = np.array(results['gnn_error'])
    gnn_errors = gnn_errors[np.isfinite(gnn_errors)]
    print(f"GNN Damping Coefficient Mean Percentage Error: {np.mean(gnn_errors):.2f}%")
    
    physics_errors = np.array(results['physics_error'])
    physics_errors = physics_errors[np.isfinite(physics_errors)]
    print(f"Physics Calculation Mean Percentage Error: {np.mean(physics_errors):.2f}%")
    
    total_errors = np.array(results['total_error'])
    total_errors = total_errors[np.isfinite(total_errors)]
    print(f"Total Error (GNN vs Reverse Calculation): {np.mean(total_errors):.2f}%")
    
    # Generate plots
    analyzer.plot_error_distributions(results)
    analyzer.plot_true_vs_predicted(results)
    
    # Analyze sensitivity to errors
    print("\nAnalyzing sensitivity to damping coefficient errors...")
    sensitivity_results = analyzer.analyze_error_sensitivity(error_range=(-50, 50), num_points=20)
    analyzer.plot_error_sensitivity(sensitivity_results)
    
    print("\nAnalysis complete. Plots saved to current directory.")

if __name__ == "__main__":
    main() 