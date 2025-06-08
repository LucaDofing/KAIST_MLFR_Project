import math, random
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.config import MAX_JOINTS
import json
import os
from glob import glob # For finding files


class FakePendulumDataset(InMemoryDataset):
    """
    Generates synthetic pendulum graph data.
    - mode="supervised": returns (x, y) where y = true damping
    - mode="unsupervised": returns (x, x_next), with no label
    """
    def __init__(self, num_graphs=1000, mode="supervised", transform=None):
        self.num_graphs = num_graphs
        self.mode = mode
        self.min_damping = 0.0
        self.max_damping = 1.0
        super().__init__('.', transform=transform)
        self.data, self.slices = self._generate()

    def _generate(self):
        if self.mode == "unsupervised":
            graphs = [self._sample_graph_unsupervised() for _ in range(self.num_graphs)]
        else:
            graphs = [self._sample_graph_supervised() for _ in range(self.num_graphs)]
        return self.collate(graphs)

    def _sample_graph_supervised(self):
        n = random.randint(1, MAX_JOINTS)
        damping = torch.empty(n).uniform_(0.05, 1.0).unsqueeze(1)

        theta = torch.empty(n).uniform_(-math.pi, math.pi)
        omega = torch.randn(n) * (1.0 - damping.squeeze()) * 3.0

        # Transform theta to sin/cos representation
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Calculate omega statistics (assuming mean=0, std=1 for this synthetic data)
        omega_mean = 0.0
        omega_std = 1.0
        omega_normalized = omega / omega_std
        
        features = torch.stack([sin_theta, cos_theta, omega_normalized], dim=1)

        if n == 1:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            send = torch.arange(n-1, dtype=torch.long)
            recv = send + 1
            edge_index = torch.cat(
                [torch.stack([send, recv], dim=0),
                 torch.stack([recv, send], dim=0)], dim=1)

        return Data(x=features, y=damping, edge_index=edge_index, 
                   omega_mean=omega_mean, omega_std=omega_std)

    def _sample_graph_unsupervised(self):
        n = random.randint(1, MAX_JOINTS)
        damping = torch.empty(n).uniform_(0.05, 1.0)

        x = []
        x_next = []
        
        # Calculate omega statistics (assuming mean=0, std=1 for this synthetic data)
        omega_mean = 0.0
        omega_std = 1.0

        for i in range(n):
            theta0 = random.uniform(-math.pi, math.pi)
            omega0 = random.uniform(-1.0, 1.0)
            theta1 = theta0 + omega0 * 0.1
            omega1 = omega0 - damping[i].item() * omega0 * 0.1

            # Transform to sin/cos representation
            sin_theta0 = math.sin(theta0)
            cos_theta0 = math.cos(theta0)
            omega0_normalized = omega0 / omega_std
            
            sin_theta1 = math.sin(theta1)
            cos_theta1 = math.cos(theta1)
            omega1_normalized = omega1 / omega_std

            x.append(torch.tensor([sin_theta0, cos_theta0, omega0_normalized]))
            x_next.append(torch.tensor([sin_theta1, cos_theta1, omega1_normalized]))

        x = torch.stack(x, dim=0)
        x_next = torch.stack(x_next, dim=0)

        if n == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            send = torch.arange(n-1, dtype=torch.long)
            recv = send + 1
            edge_index = torch.cat(
                [torch.stack([send, recv], dim=0),
                 torch.stack([recv, send], dim=0)], dim=1)

        return Data(x=x, edge_index=edge_index, x_next=x_next, 
                   omega_mean=omega_mean, omega_std=omega_std)
    

class MuJoCoPendulumDataset(InMemoryDataset):
    """
    Loads pendulum graph data from MuJoCo JSON simulation files.
    Each time step transition (t -> t+1) becomes a Data object.
    - mode="unsupervised": returns Data(x, x_next, y_true_damping, dt_step, ...)
      'y_true_damping' is the actual damping from JSON for evaluation, not direct training.
    """
    def __init__(self, root_dir, json_files_pattern="*.json", mode="unsupervised", transform=None, pre_transform=None):
        self.json_files_pattern = json_files_pattern
        self.mode = mode # "unsupervised" is the primary mode for your current setup
        # root_dir will be 'data/mujoco/'
        self.min_damping = 0.0
        self.max_damping = 1.0
        super().__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # This should point to the files in data/mujoco/raw/ if you followed typical structure
        # Or, if JSONs are directly in data/mujoco/, adjust raw_dir
        raw_path = os.path.join(self.root, "raw") # Assuming JSONs will be in data/mujoco/raw
        if not os.path.exists(raw_path):
             os.makedirs(raw_path, exist_ok=True)
             print(f"Created raw directory: {raw_path}")
             print(f"Please place your JSON and XML files from the problem description into {raw_path}")
        
        # For this setup, let's assume JSONs are directly in self.root (e.g. data/mujoco/)
        # If you move them to data/mujoco/raw/, then use os.path.join(self.raw_dir, self.json_files_pattern)
        file_paths = glob(os.path.join(self.root, self.json_files_pattern))
        return [os.path.basename(f) for f in file_paths]

    @property
    def processed_file_names(self):
        return ['mujoco_pendulum_data.pt']

    def download(self):
        # Data is provided locally, no download needed.
        # Create the 'raw' directory if it doesn't exist and files are expected there.
        # For this exercise, we assume JSONs are already in the root_dir or root_dir/raw.
        # If JSONs are in self.root, you might need to copy them to self.raw_dir
        # or adjust self.raw_paths to point to self.root.
        
        # Let's assume JSON files are in self.root (e.g. data/mujoco)
        # And we'll "copy" them conceptually to raw_paths for processing
        pass


    def process(self):
        print("Processing...")
        data_list = []
        
        # Adjusted to look for files in self.root if raw_paths is empty
        # This is a bit of a workaround for InMemoryDataset expecting raw files in self.raw_dir
        # A more robust way is to ensure your JSON files are in data/mujoco/raw/
        json_file_paths = glob(os.path.join(self.root, self.json_files_pattern))
        if not json_file_paths:
            json_file_paths = glob(os.path.join(self.raw_dir, self.json_files_pattern)) # Fallback to raw_dir

        if not json_file_paths:
            raise FileNotFoundError(f"No JSON files found matching pattern {self.json_files_pattern} in {self.root} or {self.raw_dir}")

        # First pass: collect all omega values to calculate statistics
        all_omegas = []
        
        for json_file_path in json_file_paths:
            with open(json_file_path, 'r') as f:
                sim_data = json.load(f)
                
            time_series = sim_data['time_series']
            omegas = torch.tensor(time_series['omega'], dtype=torch.float32)
            all_omegas.append(omegas)
        
        # Calculate mean and std of omega across all data
        all_omegas_tensor = torch.cat(all_omegas)
        omega_mean = all_omegas_tensor.mean().item()
        omega_std = all_omegas_tensor.std().item()
        
        print(f"Omega statistics - Mean: {omega_mean:.4f}, Std: {omega_std:.4f}")
        
        # Second pass: process data with normalized omega
        for json_file_path in json_file_paths:
            with open(json_file_path, 'r') as f:
                sim_data = json.load(f)

            metadata = sim_data['metadata']
            static_props = sim_data['static_properties']['nodes'][0] # Assuming 1 link
            time_series = sim_data['time_series']

            dt_step = torch.tensor(metadata['dt'], dtype=torch.float32)
            num_links = metadata['num_links']
            num_steps = metadata['num_steps']

            # Extract true physical parameters (for evaluation or more complex models)
            true_mass = torch.tensor([static_props['mass']], dtype=torch.float32)
            true_length = torch.tensor([static_props['length']], dtype=torch.float32)
            true_damping_coeff = torch.tensor([static_props['damping']], dtype=torch.float32)
            true_friction = torch.tensor([static_props['friction']], dtype=torch.float32)
            inertia_yy = torch.tensor([static_props['inertia']], dtype=torch.float32)
            gravity_accel = torch.tensor([metadata['gravity'][2]], dtype=torch.float32)

            
            # For a single link pendulum, edge_index is empty
            edge_index = torch.empty((2, 0), dtype=torch.long)

            thetas = torch.tensor(time_series['theta'], dtype=torch.float32)
            omegas = torch.tensor(time_series['omega'], dtype=torch.float32)
            alphas_true = torch.tensor(time_series['alpha'], dtype=torch.float32) # For future use
            torques_applied = torch.tensor(time_series['torque'], dtype=torch.float32) # For future use
            
            # Each step (t, t+1) is a sample
            for i in range(num_steps - 1):
                # Convert theta to sin/cos representation
                sin_theta_t = torch.sin(thetas[i])
                cos_theta_t = torch.cos(thetas[i])
                
                # Normalize omega
                omega_t_norm = (omegas[i] - omega_mean) / omega_std
                
                # Same for t+1
                sin_theta_t_plus_1 = torch.sin(thetas[i+1])
                cos_theta_t_plus_1 = torch.cos(thetas[i+1])
                omega_t_plus_1_norm = (omegas[i+1] - omega_mean) / omega_std
                
                # Create feature vectors - shape [1, 3]
                x_t = torch.tensor([[sin_theta_t, cos_theta_t, omega_t_norm]], dtype=torch.float32)
                x_t_plus_1 = torch.tensor([[sin_theta_t_plus_1, cos_theta_t_plus_1, omega_t_plus_1_norm]], dtype=torch.float32)
                
                # Prepare other tensors with correct shapes
                torque_t = torques_applied[i].reshape(1, 1)  # Shape [1, 1]
                alpha_t_true = alphas_true[i].reshape(1, 1)  # Shape [1, 1]

                graph_data = Data(
                    x=x_t,
                    edge_index=edge_index,
                    x_next=x_t_plus_1,
                    y_true_damping=true_damping_coeff, # Storing the physical damping
                    dt_step=dt_step,
                    # Store other potentially useful info
                    true_torque_t=torque_t, 
                    true_alpha_t=alpha_t_true,
                    true_length=true_length,
                    inertia_yy=inertia_yy,
                    gravity_accel=gravity_accel,
                    true_mass=true_mass,
                    length_com_for_gravity=true_length,
                    mass=true_mass,
                    # Store omega statistics for denormalization
                    omega_mean=torch.tensor(omega_mean, dtype=torch.float32),
                    omega_std=torch.tensor(omega_std, dtype=torch.float32),
                    file_origin=os.path.basename(json_file_path),
                    step_index_in_file=i
                )
                data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")
