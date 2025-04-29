import os
import sys
import gymnasium as gym
import numpy as np
import time
import json
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box



class Customn_link_robotEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,  # Reduced from 50 to make visualization smoother
    }

    def __init__(self, render_mode=None, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)  # 6 for arm + 2 for target
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 0.1

        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the XML from the data directory
        model_path = os.path.join(current_dir, "../../4_data/1_xml_models/n_link_robot.xml")

        super().__init__(
            model_path=model_path,
            frame_skip=5,  # Increased from 2 to make motion smoother
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )

    def save_simulation_data(self, observation, action, timestep):
        """Save simulation data to a JSON file."""
        # Create the directory if it doesn't exist
        os.makedirs("../../4_data/3_gymnasium", exist_ok=True)
        
        # Prepare data to save
        data_to_save = {
            "timestep": timestep,
            "observation": observation.tolist(),
            "action": action.tolist()
        }
        
        # Save to file
        filename = f"../../4_data/3_gymnasium/simulation_data_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(data_to_save, f, indent=4)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Save simulation data
        self.save_simulation_data(observation, action, self.data.time)
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        # Reset joint positions to random values within their ranges
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        
        # Set the state
        self.set_state(qpos, qvel)
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self):
        qpos = self.data.qpos.flat[:2].copy()  # Joint angles
        qvel = self.data.qvel.flat[:2].copy()  # Joint velocities
        
        # Convert joint angles to cos/sin representation
        cos_pos = np.cos(qpos)
        sin_pos = np.sin(qpos)
        
        # Get target position (fixed)
        target_pos = np.array([0.15, 0.15])  # Match the position in the XML
        
        return np.concatenate([cos_pos, sin_pos, qvel, target_pos])

# Register the custom environment
gym.register(
    id="Customn_link_robot-v0",
    entry_point="__main__:Customn_link_robotEnv",
    max_episode_steps=1000,
)

# Create the environment with rendering
env = gym.make("Customn_link_robot-v0", render_mode="human")

# Function to convert radians to degrees
def rad2deg(rad):
    return rad * 180.0 / np.pi

# Run the simulation
observation, info = env.reset()

print("\nTime(s) | Joint1(deg) | Joint2(deg) | Vel1(deg/s) | Vel2(deg/s) | Target_X | Target_Y | Torque1(Nm) | Torque2(Nm)")
print("-" * 100)

start_time = time.time()
try:
    for step in range(1000):
        # Random action (torques)
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Extract joint angles from observation
        # Convert from cos/sin representation to angles
        joint1_angle = np.arctan2(observation[1], observation[0])  # For joint 1
        joint2_angle = np.arctan2(observation[3], observation[2])  # For joint 2
        
        # Extract joint velocities (already in rad/s)
        joint1_vel = observation[4]
        joint2_vel = observation[5]
        
        # Extract target position
        target_x = observation[6]
        target_y = observation[7]
        
        # Convert to degrees
        joint1_angle_deg = rad2deg(joint1_angle)
        joint2_angle_deg = rad2deg(joint2_angle)
        joint1_vel_deg = rad2deg(joint1_vel)
        joint2_vel_deg = rad2deg(joint2_vel)
        
        # Get current time
        current_time = time.time() - start_time
        
        # Print the data
        print(f"{current_time:6.2f} | {joint1_angle_deg:10.2f} | {joint2_angle_deg:10.2f} | "
              f"{joint1_vel_deg:10.2f} | {joint2_vel_deg:10.2f} | {target_x:8.3f} | {target_y:8.3f} | "
              f"{action[0]:10.2f} | {action[1]:10.2f}")
        
        if terminated or truncated:
            observation, info = env.reset()
            print("\nReset Environment")
            print("-" * 100)
        
        # Small delay to make the output readable
        time.sleep(0.01)  # Reduced delay for smoother visualization
finally:
    env.close() 