import json
import matplotlib.pyplot as plt
import os
import glob

# =============================================================================
# CONFIGURATION: Change the filename here (just the filename, not full path)
# =============================================================================
# Examples from your data:

filename = "datasets/robot_L1_len0.40_rad0.075_mass3.0_ftip0.500_damp0.50_torq20.0/data/trajectory_20250608_180221_n_link_1_init_0.0_target_45.0_kp_30.0_kd_0.010_damping_0.500.json"

# =============================================================================
# AUTOMATIC PATH HANDLING - No need to change anything below
# =============================================================================
def find_data_file(filename):
    """
    Automatically searches for the data file in common data directories
    """
    # Get the directory of this script (5_plots)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Common data locations to search
    search_paths = [
        # Direct in 4_data
        os.path.join(script_dir, "..", "4_data"),
        # In 4_data/2_mujoco
        os.path.join(script_dir, "..", "4_data", "2_mujoco"),
        # In subdirectories of 4_data/2_mujoco
        os.path.join(script_dir, "..", "4_data", "2_mujoco", "*"),
    ]
    
    for search_path in search_paths:
        # Try exact path
        file_path = os.path.join(search_path, filename)
        if os.path.exists(file_path):
            return file_path
        
        # Try globbing for subdirectories
        if "*" in search_path:
            for subdir in glob.glob(search_path):
                file_path = os.path.join(subdir, filename)
                if os.path.exists(file_path):
                    return file_path
    
    return None

def list_available_files():
    """
    List all available JSON files in data directories
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.join(script_dir, "..", "4_data")
    
    json_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if file.endswith('.json'):
                # Get relative path from 4_data
                rel_path = os.path.relpath(os.path.join(root, file), data_base)
                json_files.append(rel_path)
    
    return sorted(json_files)

# 1. Try to find the JSON file
file_path = find_data_file(filename)

if file_path is None:
    print(f"❌ Could not find file: {filename}")
    print("\n📁 Available JSON files:")
    available_files = list_available_files()
    for i, file in enumerate(available_files[:10], 1):  # Show first 10
        print(f"   {i:2d}. {file}")
    if len(available_files) > 10:
        print(f"   ... and {len(available_files) - 10} more files")
    print(f"\n💡 Copy one of the filenames above and paste it into line 7 of this script")
    print(f"💡 Use just the filename (e.g., 'sim_001_n_link_1_init_90.0_target_45.0_kp_0.5_kd_0.010_damping_0.400.json')")
    exit()

print(f"✅ Loading data from: {file_path}")

with open(file_path, 'r') as f:
    data = json.load(f)

# 2. Extract time series data
# Note: Each entry in the lists is a list (e.g., [[0.1], [0.2], ...])
def flatten(series):
    return [x[0] for x in series]

time_series = data['time_series']
theta = flatten(time_series['theta'])
omega = flatten(time_series['omega'])
alpha = flatten(time_series['alpha'])
torque = flatten(time_series['torque'])

# 3. Create a time axis using dt and number of steps
dt = data['metadata']['dt']
num_steps = len(theta)
time = [i * dt for i in range(num_steps)]

# 4. Extract metadata for plot title
metadata = data.get('static_properties', {})
control_info = ""
if 'controller_gains' in metadata:
    gains = metadata['controller_gains']
    control_info = f"kp={gains.get('kp', 'N/A')}, kd={gains.get('kd', 'N/A')}"
elif 'constant_torque' in metadata:
    control_info = f"constant_torque={metadata['constant_torque']}"

plot_title = f"Robot Trajectory - {filename[:50]}{'...' if len(filename) > 50 else ''}"

# 5. Plotting
plt.figure(figsize=(12, 10))
plt.suptitle(plot_title, fontsize=12)

plt.subplot(4, 1, 1)
plt.plot(time, theta, label='Joint Angle', color='cyan', linewidth=2)
plt.ylabel('Theta (rad)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, omega, label='Angular Velocity', color='orange', linewidth=2)
plt.ylabel('Omega (rad/s)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, alpha, label='Angular Acceleration', color='green', linewidth=2)
plt.ylabel('Alpha (rad/s²)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, torque, label='Applied Torque', color='red', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Torque (N⋅m)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n📊 Plot completed for: {os.path.basename(filename)}")
print(f"📈 Simulation time: {time[-1]:.2f} seconds")
print(f"🔧 Control info: {control_info}")
