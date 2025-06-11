#!/usr/bin/env python3
"""
Trajectory Data Visualization for N-Link Robot Simulation

This script provides visualization capabilities for simulation data generated
by the MuJoCo n-link robot framework. It can automatically locate and plot
trajectory data from JSON files.

Features:
- Automatic file discovery in data directories
- Flexible file path handling
- Comprehensive trajectory plotting (position, velocity, acceleration, torque)
- Metadata extraction and display

Author: KAIST MLFR Project
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# =============================================================================
# CONFIGURATION: Change the filename here
# =============================================================================
# Examples from your data directories:
# - "trajectory_20250608_180221_n_link_1_init_0.0_target_45.0_kp_30.0_kd_0.010_damping_0.500.json"
# - "datasets/robot_L1_len0.40_rad0.075_mass3.0_ftip0.500_damp0.50_torq20.0/data/trajectory_*.json"

filename = "datasets/robot_L1_len0.40_rad0.075_mass3.0_ftip0.500_damp0.50_torq20.0/data/trajectory_20250608_180221_n_link_1_init_0.0_target_45.0_kp_30.0_kd_0.010_damping_0.500.json"

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================
PLOT_CONFIG = {
    'figure_size': (14, 12),
    'line_width': 2.0,
    'colors': {
        'theta': '#1f77b4',      # Blue
        'omega': '#ff7f0e',      # Orange
        'alpha': '#2ca02c',      # Green
        'torque': '#d62728',     # Red
    },
    'grid_alpha': 0.3,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 10,
}


class TrajectoryPlotter:
    """
    A class for plotting robot trajectory data from simulation results.
    """
    
    def __init__(self, base_data_dir: Optional[str] = None) -> None:
        """
        Initialize the trajectory plotter.
        
        Args:
            base_data_dir: Base directory for data files (default: auto-detect)
        """
        if base_data_dir:
            self.base_data_dir = Path(base_data_dir)
        else:
            # Auto-detect base data directory
            script_dir = Path(__file__).parent
            self.base_data_dir = script_dir.parent / "4_data"
        
        if not self.base_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.base_data_dir}")
    
    def find_data_file(self, filename: str) -> Optional[Path]:
        """
        Automatically search for the data file in common data directories.
        
        Args:
            filename: Name or relative path of the data file
            
        Returns:
            Path to the found file, or None if not found
        """
        # Search paths in order of preference
        search_paths = [
            self.base_data_dir,
            self.base_data_dir / "2_mujoco",
            self.base_data_dir / "2_mujoco" / "datasets",
        ]
        
        for search_path in search_paths:
            # Try exact path
            file_path = search_path / filename
            if file_path.exists():
                return file_path
            
            # Try recursive search for pattern matching
            if "*" in filename or "?" in filename:
                matches = list(search_path.rglob(filename))
                if matches:
                    return matches[0]  # Return first match
            else:
                # Search recursively for exact filename
                matches = list(search_path.rglob(filename))
                if matches:
                    return matches[0]
        
        return None
    
    def list_available_files(self, max_files: int = 20) -> List[str]:
        """
        List all available JSON files in data directories.
        
        Args:
            max_files: Maximum number of files to return
            
        Returns:
            List of relative file paths
        """
        json_files = []
        
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                if file.endswith('.json') and 'trajectory' in file:
                    # Get relative path from base data directory
                    rel_path = Path(root).relative_to(self.base_data_dir) / file
                    json_files.append(str(rel_path))
        
        return sorted(json_files)[:max_files]
    
    def load_trajectory_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Load trajectory data from JSON file.
        
        Args:
            file_path: Path to the JSON data file
            
        Returns:
            Dictionary containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            KeyError: If required data fields are missing
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
        
        # Validate required fields
        required_fields = ['time_series', 'metadata']
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field '{field}' in data file")
        
        return data
    
    def extract_time_series(self, data: Dict[str, Any]) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        Extract and flatten time series data.
        
        Args:
            data: Loaded trajectory data
            
        Returns:
            Tuple of (time_array, trajectory_data)
        """
        time_series = data['time_series']
        
        # Flatten nested lists (each entry is typically [[value]])
        def flatten(series: List[List[float]]) -> List[float]:
            return [x[0] if isinstance(x, list) and len(x) > 0 else x for x in series]
        
        # Extract trajectory data
        trajectory_data = {}
        for key in ['theta', 'omega', 'alpha', 'torque']:
            if key in time_series:
                trajectory_data[key] = flatten(time_series[key])
        
        # Create time axis
        dt = data['metadata'].get('dt', 0.01)
        num_steps = len(trajectory_data.get('theta', []))
        time_array = [i * dt for i in range(num_steps)]
        
        return time_array, trajectory_data
    
    def extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant metadata for plot titles and information.
        
        Args:
            data: Loaded trajectory data
            
        Returns:
            Dictionary containing relevant metadata
        """
        metadata = data.get('metadata', {})
        static_props = data.get('static_properties', {})
        
        # Extract control information
        control_info = ""
        if 'controller_gains' in static_props:
            gains = static_props['controller_gains']
            kp = gains.get('kp', 'N/A')
            kd = gains.get('kd', 'N/A')
            control_info = f"kp={kp}, kd={kd}"
        elif 'constant_torque' in static_props:
            control_info = f"constant_torque={static_props['constant_torque']}"
        
        return {
            'num_links': metadata.get('num_links', 'N/A'),
            'simulation_time': metadata.get('simulation_time', 'N/A'),
            'dt': metadata.get('dt', 'N/A'),
            'control_info': control_info,
            'initial_angle': static_props.get('initial_angle_deg', 'N/A'),
            'target_angle': static_props.get('target_angle_deg', 'N/A'),
        }
    
    def create_plot(
        self, 
        time_array: List[float], 
        trajectory_data: Dict[str, List[float]], 
        metadata: Dict[str, Any], 
        filename: str
    ) -> None:
        """
        Create comprehensive trajectory plot.
        
        Args:
            time_array: Time points
            trajectory_data: Dictionary of trajectory data arrays
            metadata: Metadata for plot titles
            filename: Original filename for plot title
        """
        # Create figure with subplots
        fig = plt.figure(figsize=PLOT_CONFIG['figure_size'])
        gs = gridspec.GridSpec(4, 1, hspace=0.3)
        
        # Create main title
        title = f"Robot Trajectory Analysis"
        if len(filename) > 60:
            title += f"\n{filename[:60]}..."
        else:
            title += f"\n{filename}"
        
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')
        
        # Plot data
        plot_configs = [
            ('theta', 'Joint Angle', 'θ (rad)', 0),
            ('omega', 'Angular Velocity', 'ω (rad/s)', 1),
            ('alpha', 'Angular Acceleration', 'α (rad/s²)', 2),
            ('torque', 'Applied Torque', 'τ (N⋅m)', 3),
        ]
        
        for data_key, title, ylabel, subplot_idx in plot_configs:
            ax = fig.add_subplot(gs[subplot_idx])
            
            if data_key in trajectory_data:
                ax.plot(
                    time_array, 
                    trajectory_data[data_key],
                    label=title,
                    color=PLOT_CONFIG['colors'][data_key],
                    linewidth=PLOT_CONFIG['line_width']
                )
                
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_fontsize'])
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(fontsize=PLOT_CONFIG['legend_fontsize'])
                
                # Add statistics as text
                data_array = trajectory_data[data_key]
                if data_array:
                    stats_text = (
                        f"Min: {min(data_array):.3f}, "
                        f"Max: {max(data_array):.3f}, "
                        f"Mean: {sum(data_array)/len(data_array):.3f}"
                    )
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f"No {data_key} data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_fontsize'])
        
        # Set x-label only on bottom subplot
        fig.axes[-1].set_xlabel('Time (s)', fontsize=PLOT_CONFIG['label_fontsize'])
        
        # Add metadata text box
        metadata_text = (
            f"Links: {metadata['num_links']} | "
            f"dt: {metadata['dt']} s | "
            f"Duration: {metadata['simulation_time']} s\n"
            f"Initial: {metadata['initial_angle']}° | "
            f"Target: {metadata['target_angle']}° | "
            f"{metadata['control_info']}"
        )
        
        fig.text(0.02, 0.02, metadata_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_trajectory(self, filename: str) -> None:
        """
        Main method to plot trajectory data from file.
        
        Args:
            filename: Name or path of the data file to plot
        """
        # Find the data file
        file_path = self.find_data_file(filename)
        
        if file_path is None:
            print(f"❌ Could not find file: {filename}")
            print("\n📁 Available JSON files:")
            available_files = self.list_available_files()
            
            for i, file in enumerate(available_files, 1):
                print(f"   {i:2d}. {file}")
            
            if len(available_files) == 0:
                print("   No trajectory files found!")
            else:
                print(f"\n💡 Copy one of the filenames above and update line 21 in this script")
            return
        
        print(f"✅ Loading data from: {file_path}")
        
        try:
            # Load and process data
            data = self.load_trajectory_data(file_path)
            time_array, trajectory_data = self.extract_time_series(data)
            metadata = self.extract_metadata(data)
            
            # Create plot
            self.create_plot(time_array, trajectory_data, metadata, filename)
            
            # Print summary
            print(f"\n📊 Plot completed for: {file_path.name}")
            print(f"📈 Simulation time: {time_array[-1]:.2f} seconds" if time_array else "📈 No time data")
            print(f"🔧 Control info: {metadata['control_info']}")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Error loading data: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main() -> None:
    """Main entry point for the plotting script."""
    try:
        plotter = TrajectoryPlotter()
        plotter.plot_trajectory(filename)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
