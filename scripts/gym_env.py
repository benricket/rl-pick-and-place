import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from mujoco_example import ee_to_block_pos, set_joints, get_joints
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class ArmEnv(gym.Env):
    """
    Class to implement the Kinova pick-and-place task as an RL environment
    """
    def __init__(self):
        # Initialize environment

        # Holds pose of end effector, target (block to pick up) and goal (block to set the target on)
        self.observation_space = gym.spaces.Dict(
            {
                "ee_pos": gym.spaces.Box(-5, 5, shape=(3,), dtype=float),  # x,y,z
                #"ee_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                "target_pos": gym.spaces.Box(-5, 5, shape=(3,), dtype=float),  # x,y,z
                #"target_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                #"goal_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                #"goal_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                "joint_pos": gym.spaces.Box(-np.pi, np.pi, shape=(7,), dtype=float)
            }
        )

        # We output a desired pose, which the handling of the action will approximate given dt
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),   # 7 joints + 1 gripper
            dtype=np.float32,
        )

        self.model = mujoco.MjModel.from_xml_path('../kinova_gen3/rl_scene.xml')
        self.data = mujoco.MjData(self.model)
        self.iter_count = 0

        # attributes for reward func calculations
        self.last_ee_to_block_dist = ee_to_block_pos(self.model,self.data)

    def _get_obs(self):
        # Maps environment state to the returned observation
        obs = {}
        block_name = 'cube'
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
        block_pos = self.data.geom_xpos[block_id].copy()

        ee_name = "pinch_site"
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        ee_pos = self.data.site_xpos[ee_id].copy()

        obs["ee_pos"] = ee_pos
        obs["target_pos"] = block_pos
        obs["joint_pos"] = get_joints(self.model,self.data)
        
        return obs

    def _get_info(self):
        # gets auxiliary info for debugging
        return {}

    def _compute_reward(self):
        reward = 0.0

        # reward getting closer to the block
        dist = ee_to_block_pos(self.model,self.data)
        dist_change = dist - self.last_ee_to_block_dist
        reward += -10 * dist_change
        self.last_ee_to_block_dist = dist

        reward = -dist

        return reward

    def _is_truncated(self):
        return False
    
    def _is_terminated(self):
        dist = ee_to_block_pos(self.model,self.data)
        if dist < 0.2:
            return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Starts a new episode
        """
        super().reset(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs,info
    
    def step(self, action):
        """
        Update step of the simulation with the given action
        """
        reward = 0
        obs = 0
        terminated = False
        truncated = False
        info = {}

        # Apply action
        joint_ctrl = action[0:7]
        joint_ctrl *= np.pi # map -1,1 to -pi,pi
        gripper_ctrl = action[7]
        gripper_ctrl = np.interp(gripper_ctrl, [-1,1], [0,255])

        curr_joint_vals = get_joints(self.model,self.data)
        target_joint_vals = curr_joint_vals + joint_ctrl * 0.05
        set_joints(self.model,self.data,target_joint_vals,0.0)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Get reward
        reward = self._compute_reward()

        # Check termination
        truncated = self._is_truncated()
        terminated = self._is_terminated()
        if terminated:
            reward += 10000

        self.iter_count += 1

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":

    register(
        id="KinovaEnv",
        entry_point="gym_env:ArmEnv",
        max_episode_steps=200,
    )

    env = gym.make("KinovaEnv")
    check_env(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="../logs/ppo_kinova/"
    )

    model.learn(total_timesteps=25_000)
    model.save("kinova_test")


