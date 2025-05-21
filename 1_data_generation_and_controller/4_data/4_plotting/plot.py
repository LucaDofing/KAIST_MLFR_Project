import json
import matplotlib.pyplot as plt

path_to_mujoco= "/home/michael/Documents/KAIST/KAIST_MLFR_Project/KAIST_MLFR_Project/1_data_generation_and_controller/4_data/2_mujoco/"
# 1. Load the JSON file

abs_path=path_to_mujoco + "1_link_pd_controller"+".json"

with open(abs_path, 'r') as f:
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

# 4. Plotting
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time, theta, label='Theta', color='cyan')
plt.ylabel('Theta')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, omega, label='Omega', color='orange')
plt.ylabel('Omega')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, alpha, label='Alpha', color='green')
plt.ylabel('Alpha')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, torque, label='Torque', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Torque')
plt.legend()

plt.tight_layout()
plt.show()
