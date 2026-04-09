import numpy as np
from typing import Optional

import gymnasium as gym

class ArmEnv(gym.Env):
    """
    Class to implement the Kinova pick-and-place task as an RL environment
    """
    def __init__(self):
        # Initialize environment

        # Holds pose of end effector, target (block to pick up) and goal (block to set the target on)
        self.observation_space = gym.spaces.Dict(
            {
                "ee_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                "ee_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                "target_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                "target_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                "goal_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                "goal_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
            }
        )

        # We output a desired pose, which the handling of the action will approximate given dt
        self.action_space = gym.spaces.Dict(
            {
                "command_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                "command_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
            }
        )

    def _get_obs(self):
        # Maps environment state to the returned observation
        pass

    def _get_info(self):
        # gets auxiliary info for debugging
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Starts a new episode
        """
        super().reset(seed=seed)
    
    def step(self, action):
        """
        Update step of the simulation with the given action
        """
        pass




