from gymnasium.envs.registration import register
from .run_reacher_data import CustomReacherEnv

register(
    id="CustomReacher-v0",
    entry_point="gym_minimal.run_reacher_data:CustomReacherEnv",
    max_episode_steps=1000,
) 