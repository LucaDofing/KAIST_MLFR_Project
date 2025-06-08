import torch
from torch import nn
from torch_geometric.nn import GCNConv
from src.config import INPUT_DIM

class DampingGCN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),  # [sin(theta), cos(theta), normalized_omega]
            nn.Sigmoid()
        )
        
        # Physics processing with more capacity
        self.physics_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [alpha, torque]
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Graph convolutions for state
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Graph convolutions for physics
        self.physics_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.physics_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Combined processing
        self.combined_conv = GCNConv(hidden_dim * 2, hidden_dim)
        
        # Final prediction with more layers
        self.damping_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Physical constraints
        self.min_damping = 0.0
        self.max_damping = 1.0
        
    def forward(self, data):
        # Extract features
        x = data.x  # [sin(theta), cos(theta), normalized_omega]
        alpha = data.true_alpha_t  # angular acceleration
        torque = data.true_torque_t  # applied torque
        edge_index = data.edge_index
        
        # Process state information
        state_features = self.state_encoder(x)
        state_features = torch.relu(self.conv1(state_features, edge_index))
        state_features = torch.relu(self.conv2(state_features, edge_index))
        
        # Process physics information
        physics_input = torch.cat([alpha, torque], dim=1)
        physics_features = self.physics_encoder(physics_input)
        physics_features = torch.relu(self.physics_conv1(physics_features, edge_index))
        physics_features = torch.relu(self.physics_conv2(physics_features, edge_index))
        
        # Combine state and physics information
        combined_features = torch.cat([state_features, physics_features], dim=1)
        combined_features = torch.relu(self.combined_conv(combined_features, edge_index))
        
        # Predict damping with physical constraints
        damping = self.damping_predictor(combined_features)
        damping = torch.sigmoid(damping) * (self.max_damping - self.min_damping) + self.min_damping
        
        return damping
    
    def compute_next_state(self, data, damping):
        """Compute next state using full physical simulation"""
        # Extract original theta and omega from sin(theta) and cos(theta)
        theta = torch.atan2(data.x[:, 0], data.x[:, 1])  # atan2(sin(theta), cos(theta))
        omega = data.x[:, 2] * data.omega_std + data.omega_mean  # Denormalize omega
        
        dt = data.dt_step
        mass = data.true_mass
        length = data.true_length
        inertia = data.inertia_yy
        g = data.gravity_accel
        applied_torque = data.true_torque_t
        
        # Compute torques
        torque_gravity = -mass * g * length * torch.sin(theta)
        torque_damping = -damping.squeeze(-1) * omega
        net_torque = applied_torque.squeeze(-1) + torque_gravity + torque_damping
        
        # Compute angular acceleration
        alpha = net_torque / inertia
        
        # Update state using semi-implicit Euler
        omega_next = omega + alpha * dt
        theta_next = theta + omega_next * dt
        
        # Convert back to sin(theta), cos(theta), normalized_omega format
        sin_theta_next = torch.sin(theta_next)
        cos_theta_next = torch.cos(theta_next)
        omega_next_norm = (omega_next - data.omega_mean) / data.omega_std
        
        return torch.stack([sin_theta_next, cos_theta_next, omega_next_norm], dim=1)
