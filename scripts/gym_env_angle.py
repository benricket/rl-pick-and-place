"""
Defines the RL environment for the Kinova arm taking actions in joint space
"""
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from scipy.spatial.transform import Rotation
from mujoco_example import ee_to_block_pos, set_joints, get_joints, get_gripper_open_close, get_joint_velocities, get_joint_limits


SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = (SCRIPT_DIR.parent / "rl_environment.xml").resolve()

FIND_OBJECT = 0
FIND_GOAL = 1

def rotm_to_euler(rotm):
    """
    Converts a rotation matrix to Euler angles in radians
    Args:
        rotm (np.ndarray): rotation matrix input

    Returns: 
        Array of Euler angles in radians
    """
    rot = Rotation.from_matrix(rotm)
    return rot.as_euler('zxy',degrees=False)

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Defines a linear learning rate schedule.

    Args: 
        initial_value (float): Initial learning rate
        final_value (float): Final learning rate

    Returns:
        A callable function that computes the learning rate given the proportion 
        progress remaining as a float
    """
    def func(progress_remaining: float) -> float:
        """
        Function mapping progress remaining to learning rate

        Args: 
            progress_remainin (float): proportion of progress remaining
        Returns: current learning rate
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

class ArmEnv(gym.Env):
    """
    Class to implement the Kinova pick-and-place task as an RL environment
    """
    def __init__(self):
        # Initialize environment

        # Define bounds for observation space
        low = np.concatenate([
            np.full(3, -5,dtype=np.float32),          # ee_pos
            np.full(3, -np.pi,dtype=np.float32),      # ee_rot
            np.full(3, -5, dtype=np.float32),          # target_pos
            np.full(6, -np.pi, dtype=np.float32),      # joint_pos
            np.full(6, -np.pi, dtype=np.float32),          # joint to min
            np.full(6, -np.pi, dtype=np.float32),         # joint to max
            np.full(6, -10, dtype=np.float32),         # joint vel
            np.array([0.0], dtype=np.float32),          # gripper
            np.array([0.0], dtype=np.float32),          # at_box
        ])

        high = np.concatenate([
            np.full(3, 5,dtype=np.float32),          # ee_pos
            np.full(3, np.pi,dtype=np.float32),      # ee_rot
            np.full(3, 5, dtype=np.float32),          # target_pos
            np.full(6, np.pi, dtype=np.float32),      # joint_pos
            np.full(6, np.pi, dtype=np.float32),          # joint to min
            np.full(6, np.pi, dtype=np.float32),         # joint to max
            np.full(6, 10, dtype=np.float32),         # joint vel
            np.array([1.0], dtype=np.float32),          # gripper
            np.array([1.0], dtype=np.float32),          # at_box
        ])

        self.random_level = 0 # how large the space is that the target box can be spawned in
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # We output a desired pose, which the handling of the action will approximate given dt
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),   # 6 joints + 1 gripper
            dtype=np.float32,
        )

        # Initialize simulation
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.iter_count = 0
        self.state = FIND_OBJECT

        self.q_min,self.q_max = get_joint_limits()

        # attributes for reward func calculations
        self.last_ee_to_block_dist = np.linalg.norm(ee_to_block_pos(self.model,self.data))
        self.ee_alignment = 0.0
        self.ee_rotm = np.eye(3).flatten()
        self.gripper = 0.0
        self.joint_ctrl = np.zeros(shape=(6,))
        print(self.ee_rotm)
    
    def set_random_level(self, level):
        """
        Controls how difficult the task should be (how much the block can be moved)
        0 means the block will be in the same place, 1 means the block will be between 
        [0,2pi] angle and [0.3,0.7] distance, with linear interpolation in between.

        Args:
            level (float): float to set the random level to, [0,1]
        """
        self.random_level = level

    def _get_obs(self):
        """
        Return an observation from the environment
        """
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

        gripper = get_gripper_open_close(self.model, self.data)

        self.ee_rotm = ee_rotm
        self.gripper = gripper

        # distance to joint limits
        q = get_joints(self.model,self.data)
        q_to_min = q - self.q_min
        q_to_max = self.q_max - q

        return np.concatenate([
            ee_pos, # position of EE
            ee_rot, # rotation matrix of EE
            block_pos, # block position in Cartesian coords
            get_joints(self.model,self.data), # joint angles
            q_to_min, # how far joint angles are from min
            q_to_max, # how far joint angles are from max
            get_joint_velocities(self.model,self.data), # joint velocities
            np.array([gripper]), # if the gripper is closed or open
            np.array([int(self.last_ee_to_block_dist < 0.03)]) # if we're very near the block
        ]).astype(np.float32)

    def _get_info(self):
        # gets auxiliary info for debugging (nothing yet)
        return {}

    def _compute_reward(self):
        """
        Get the reward from a given state

        Returns:
            reward (float) judging the value of the environment state after the last action
        """
        reward = 0.0
        info = {}

        # reward getting closer to the block
        dist = ee_to_block_pos(self.model,self.data) # on the order of 0.4
        dist_norm = np.linalg.norm(dist)
        
        progress_reward = 100.0 * (self.last_ee_to_block_dist - dist_norm)
        reward += progress_reward
        info["rew_progress"] = progress_reward

        dist_reward = -2.5 * dist_norm
        reward += dist_reward
        info["rew_dist"] = dist_reward

        vel_reward = -0.05 * np.linalg.norm(self.joint_ctrl)**2
        reward += vel_reward
        info["rew_vel_sq"] = vel_reward

        self.last_ee_to_block_dist = dist_norm
        rot_z = self.ee_rotm[:,2] # get third column
        cos_sim = np.dot(rot_z,dist) / dist_norm
        self.ee_alignment = cos_sim
        align_reward = 0.03 * cos_sim
        reward += align_reward
        info["rew_align"] = align_reward

        # penalize being near joint limits
        lim_margin = 0.15  # radians

        q = get_joints(self.model,self.data)
        q_to_min = q - self.q_min
        q_to_max = self.q_max - q

        lower_violation = np.maximum(0.0, lim_margin - q_to_min)
        upper_violation = np.maximum(0.0, lim_margin - q_to_max)

        joint_limit_penalty = -0.05 * np.sum(lower_violation**2 + upper_violation**2) * 1
        reward += joint_limit_penalty

        # Boost reward near the block --- disabled for now
        # if dist_norm < 0.3:
        #     reward += 0.01
        #     if dist_norm < 0.1:
        #         reward += 0.3
        #         if dist_norm < 0.03:
        #             reward += 3

        # if self.state == FIND_GOAL:
        #     reward += 30 * self.gripper
        gripper_reward = 0.0
        if self.state == FIND_OBJECT:
            gripper_reward = -0.1 * self.gripper * 0
        reward += gripper_reward
        info["rew_gripper"] = gripper_reward

        # if dist_norm < 0.10:
        #     reward += 1.0
        # if dist_norm < 0.05:
        #     reward += 3.0
        # if dist_norm < 0.02:
        #     reward += 10.0
        # if dist_norm < 0.01:
        #     reward += 30.0

        return reward,info

    def _is_truncated(self):
        return False
    
    def _is_terminated(self):
        """
        Return True if the block is close enough (within 0.05)
        """
        dist_norm = self.last_ee_to_block_dist
        if dist_norm < 0.05:
            return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Starts a new episode, resets the simulation
        """
        super().reset(seed=seed)
        self.iter_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # Use gymnasium random for setting the block position
        # This behaves properly with the environment seed
        block_angle = self.np_random.uniform(0, 2*np.pi*self.random_level)
        block_r = self.np_random.uniform(
            0.5 - 0.2*self.random_level,
            0.5 + 0.2*self.random_level
        )

        block_x = block_r * np.cos(block_angle)
        block_y = block_r * np.sin(block_angle)
        
        cube_jnt_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_free"
        )

        qpos_addr = self.model.jnt_qposadr[cube_jnt_id]

        self.data.qpos[qpos_addr:qpos_addr + 7] = np.array([
            block_x, block_y, 0.03,   # x, y, z
            1.0, 0.0, 0.0, 0.0        # quaternion: qw, qx, qy, qz
        ])

        mujoco.mj_forward(self.model, self.data)
        self.last_ee_to_block_dist = np.linalg.norm(ee_to_block_pos(self.model,self.data))
        self.gripper = 0.0
        self.state = FIND_OBJECT
        self.ee_alignment = 0.0
        self.joint_ctrl = np.zeros(shape=(6,))

        self.q_min = self.model.jnt_range[:6, 0]
        self.q_max = self.model.jnt_range[:6, 1]

        obs = self._get_obs()
        info = self._get_info()
        return obs,info
    
    def step(self, action):
        """
        Update step of the simulation with the given action

        Args:
            action (np.ndarray): 7-element array of 6 joint changes plus a gripper control
                scaled from -1 to 1
        """
        reward = 0
        obs = 0
        terminated = False
        truncated = False

        max_joint_speed = 3.0 # rad/s, taken roughly from Kinova docs
        control_dt = self.model.opt.timestep * 5 # time per each model step
        max_joint_change = max_joint_speed * control_dt

        # Apply action
        joint_ctrl = action[0:6]
        joint_ctrl *= max_joint_change # map -1,1 to -pi,pi
        gripper_ctrl = action[6]
        gripper_ctrl = np.interp(gripper_ctrl, [-1,1], [0,1])

        ##### HARDCODED GRIPPER OPEN
        gripper_ctrl = 1

        curr_joint_vals = get_joints(self.model,self.data)
        target_joint_vals = curr_joint_vals + joint_ctrl
        self.joint_ctrl = joint_ctrl
        set_joints(self.model,self.data,target_joint_vals,gripper_ctrl)

        # Step simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Get reward
        reward,info = self._compute_reward()

        # Reward for termination, based on how early it occurs
        term_reward = 0.0
        if self.state == FIND_OBJECT and self.last_ee_to_block_dist < 0.05:
            term_reward = 500.0 * (800 - self.iter_count) / 800
        reward += term_reward
        info["rew_at_target"] = term_reward
        #     self.state = FIND_GOAL

        # Check termination
        truncated = self._is_truncated()
        terminated = self._is_terminated()

        self.iter_count += 1

        return obs, reward, terminated, truncated, info

class RewardPrintCallback(BaseCallback):
    """
    Callback for logging custom metrics during training
    """
    def __init__(self, keys=None, print_every=3_200, verbose=0):
        """
        Initialize the callback

        Args:
            keys (dict): What keys in info{} to log the value of 
            print_every (int): How many timesteps to print logged values after
        """
        super().__init__(verbose)
        self.keys = keys
        self.buffer = {}
        self.last_print_step = 0
        self.print_every = print_every

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        # Log all specified keys into the buffer
        for info in infos:
            for k in self.keys:
                if k in info:
                    self.buffer.setdefault(k, []).append(info[k])

        return True

    def _on_rollout_end(self) -> None:
        # Print values if enough timesteps have elapsed
        if self.num_timesteps - self.last_print_step < self.print_every:
            return
        print("\n[Reward terms over rollout]")
        means = {}
        for k, vals in self.buffer.items():
            means[k] = np.mean(vals)
        abs_rew_mean = np.sum([np.abs(val) for val in means.values()])
        for k, m in means.items():
            print(f"{k}: mean={m: .4f}, proportion={np.abs(m)/abs_rew_mean: .4f}")
            self.logger.record(f"reward_terms/{k}_mean", m)
            self.logger.record(f"reward_terms/{k}_proportion", np.abs(m)/abs_rew_mean)
        print("-" * 30)

        self.buffer.clear()
        self.last_print_step = self.num_timesteps


if __name__ == "__main__":
    now = datetime.now()
    time_str = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=1200,
    )
    env = gym.make("KinovaEnv")
    check_env(env)

    env = DummyVecEnv([lambda: Monitor(gym.make("KinovaEnv"))])    
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Define PPO parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=0.0004,
        ent_coef=0.0,
        clip_range=0.1,
        tensorboard_log=f"../logs/ppo_kinova_{time_str}_curriculum_smalldelta_2x256/",
        #n_steps=2048,
    )

    cb_keys = ["rew_dist","rew_vel_sq","rew_align","rew_gripper","rew_progress","rew_at_target"]
    log_name = "ppo_curriculum_progress_weight"
    try:
        env.env_method("set_random_level", 0)
        model.learn(total_timesteps=800_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name)
        env.env_method("set_random_level", 0.02)
        model.learn(total_timesteps=800_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.env_method("set_random_level", 0.07)
        model.learn(total_timesteps=500_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.env_method("set_random_level", 0.15)
        model.learn(total_timesteps=500_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.env_method("set_random_level", 0.25)
        model.learn(total_timesteps=500_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.env_method("set_random_level", 0.5)
        model.learn(total_timesteps=500_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Interrupted; saving now...")
    finally:
        model.save(f"kinova_{time_str}")
        env.save(f"vecnorm_{time_str}.pkl")


