import numpy as np
from pathlib import Path
from typing import Optional, Callable

import gymnasium as gym
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from mujoco_example import ee_to_block_pos, set_joints, get_joints
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = (SCRIPT_DIR.parent / "rl_environment.xml").resolve()

def rotm_to_euler(rotm):
    rot = Rotation.from_matrix(rotm)
    return rot.as_euler('zxy',degrees=False)

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    param initial_value: Initial learning rate.
    return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        param progress_remaining:
        return: current learning rate
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

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
                "ee_rot": gym.spaces.Box(-np.pi, np.pi, shape=(3,), dtype=float),  # rot_z,rot_x,rot_y
                "target_pos": gym.spaces.Box(-5, 5, shape=(3,), dtype=float),  # x,y,z
                #"target_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                #"goal_pos": gym.spaces.Box(-1, 1, shape=(3,), dtype=float),  # x,y,z
                #"goal_rot": gym.spaces.Box(0, 2*np.pi, shape=(3,), dtype=float),  # roll,pitch,yaw
                "joint_pos": gym.spaces.Box(-np.pi, np.pi, shape=(6,), dtype=float)
            }
        )

        # We output a desired pose, which the handling of the action will approximate given dt
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),   # 7 joints + 1 gripper
            dtype=np.float32,
        )

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.iter_count = 0

        # attributes for reward func calculations
        self.last_ee_to_block_dist = np.linalg.norm(ee_to_block_pos(self.model,self.data))
        self.ee_rotm = np.eye(3).flatten()
        print(self.ee_rotm)

    def _get_obs(self):
        # Maps environment state to the returned observation
        obs = {}
        block_name = 'cube'
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
        block_pos = self.data.geom_xpos[block_id].copy()

        ee_name = "pinch_site"
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        ee_pos = self.data.site_xpos[ee_id].copy()

        ee_rotm = self.data.site_xmat[ee_id].copy().reshape(3,3)
        ee_rot = rotm_to_euler(ee_rotm)

        obs["ee_pos"] = ee_pos
        obs["target_pos"] = block_pos
        obs["joint_pos"] = get_joints(self.model,self.data)
        obs["ee_rot"] = ee_rot

        self.ee_rotm = ee_rotm
        return obs

    def _get_info(self):
        # gets auxiliary info for debugging
        return {}

    def _compute_reward(self):
        reward = 0.0

        # reward getting closer to the block
        dist = ee_to_block_pos(self.model,self.data) # on the order of 0.4
        dist_norm = np.linalg.norm(dist)
        
        reward += 20.0 * (self.last_ee_to_block_dist - dist_norm)
        reward += -2.0 * dist_norm
        self.last_ee_to_block_dist = dist_norm

        rot_z = self.ee_rotm[:,2] # get third column
        cos_sim = np.dot(rot_z,dist) / dist_norm
        reward += 1.0 * cos_sim

        #reward = 1.0 * cos_sim

        return reward

    def _is_truncated(self):
        return False
    
    def _is_terminated(self):
        dist_norm = np.linalg.norm(ee_to_block_pos(self.model,self.data))
        if dist_norm < 0.1:
            return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Starts a new episode
        """
        super().reset(seed=seed)
        self.iter_count = 0
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.last_ee_to_block_dist = np.linalg.norm(ee_to_block_pos(self.model,self.data))
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

        max_joint_change = 0.2 # rad

        # Apply action
        joint_ctrl = action[0:6]
        joint_ctrl *= max_joint_change # map -1,1 to -pi,pi
        gripper_ctrl = action[6]
        gripper_ctrl = np.interp(gripper_ctrl, [-1,1], [0,255])

        curr_joint_vals = get_joints(self.model,self.data)
        target_joint_vals = curr_joint_vals + joint_ctrl
        set_joints(self.model,self.data,target_joint_vals,0.0)

        # Step simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Get reward
        reward = self._compute_reward()

        # Check termination
        truncated = self._is_truncated()
        terminated = self._is_terminated()
        if terminated:
            reward += 2000.0 * (400 - self.iter_count) / 400

        self.iter_count += 1

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":

    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=400,
    )

    env = gym.make("KinovaEnv")
    check_env(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(0.005,0.003),
        #learning_rate=0.001,
        tensorboard_log="../logs/ppo_kinova/"
    )

    try:
        model.learn(total_timesteps=500_000)
    except KeyboardInterrupt:
        print("Interrupted; saving now...")
    finally:
        model.save("kinova_test_angle")


