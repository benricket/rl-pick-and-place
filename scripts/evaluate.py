import time
import sys
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from gym_env import ArmEnv
import numpy as np
import matplotlib.pyplot as plt
from mujoco_example import get_joint_velocities
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def get_distance(m,d):
    pass

def get_alignment(m,d):
    pass

def main(use_gui: bool):
    """
    Script for evaluating the performance of a learned policy
    """
        
    NUM_STEPS = 1200
    NUM_TRIALS = 10

    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=NUM_STEPS+1,
    )

    # Titles of models to learn and their file name
    learned_models = {
        "3.2M steps, 3x512 size, curriculum": "kinova_2026-5-1-23-14-8",
        "1.7M steps, 3x512 size, curriculum": "kinova_2026-5-1-16-13-25",
        "vectorized": "kinova_2026-5-4-21-35-42"}

    dict_dist = {}
    dict_align = {}
    dict_jerk = {}

    # Define arrays to hold logged values in

    for m_name,m_filename in learned_models.items():

        raw_env = DummyVecEnv([lambda: Monitor(gym.make("KinovaEnv"))])
        raw_env.env_method("set_random_level", 0.5)

        vecnorm_path = Path(f"vecnorm_{m_filename[7:]}.pkl") # hacky fix to deal current naming scheme

        try:
            env = VecNormalize.load(str(vecnorm_path), raw_env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded VecNormalize: {vecnorm_path}")
        except FileNotFoundError:
            env = raw_env
            print(f"No VecNormalize found for {m_filename}; using raw observations")

        base_env = raw_env.envs[0].unwrapped
        model = PPO.load(m_filename)

        arr_dist = np.zeros((NUM_TRIALS,NUM_STEPS))
        arr_align = np.zeros((NUM_TRIALS,NUM_STEPS))
        arr_jerk = np.zeros((NUM_TRIALS,NUM_STEPS - 2))

        for i in range(NUM_TRIALS):
            print(f"starting trial {i}")
            obs = env.reset()
            arr_vel = np.zeros((6,NUM_STEPS))
            for j in range(NUM_STEPS):
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                info = infos[0]

                vel = get_joint_velocities(base_env.model,base_env.data)
                arr_vel[:,j] = vel
                arr_dist[i,j] = base_env.last_ee_to_block_dist
                arr_align[i,j] = base_env.ee_alignment

                if done:
                    arr_vel[:,j+1:] = 0.0
                    arr_dist[i,j+1:] = 0.0
                    arr_align[i,j+1:] = 1.0
                    break
            
            # Get jerk from velocity
            dt = 1
            arr_jerk_raw = np.diff(arr_vel,n=2) / dt
            arr_jerk[i,:] = np.linalg.norm(arr_jerk_raw,axis=0)
        
        arr_dist_mean = np.mean(arr_dist,axis=0)
        arr_dist_p10 = np.quantile(arr_dist,0.1,axis=0)
        arr_dist_p90 = np.quantile(arr_dist,0.9,axis=0)
        arr_jerk_mean = np.mean(arr_jerk,axis=0)
        arr_jerk_p10 = np.quantile(arr_jerk,0.1,axis=0)
        arr_jerk_p90 = np.quantile(arr_jerk,0.9,axis=0)
        arr_align_mean = np.mean(arr_align,axis=0)
        arr_align_p10 = np.quantile(arr_align,0.1,axis=0)
        arr_align_p90 = np.quantile(arr_align,0.9,axis=0)

        dict_dist[m_name] = (arr_dist_mean,arr_dist_p10,arr_dist_p90)
        dict_jerk[m_name] = (arr_jerk_mean,arr_jerk_p10,arr_jerk_p90)
        dict_align[m_name] = (arr_align_mean,arr_align_p10,arr_align_p90)

    fig,ax = plt.subplots(3,1)
    for m_name,vals in dict_dist.items():
        ax[0].plot(np.arange(NUM_STEPS),vals[0],label=m_name)
        ax[0].fill_between(np.arange(NUM_STEPS),vals[1],vals[2],alpha=0.5)
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Distance from target (m)")
    ax[0].set_title("Distance from target over time")
    ax[0].hlines(0.05,0,NUM_STEPS,colors='k',linestyle="dashed")
    ax[0].legend()

    for m_name,vals in dict_align.items():
        ax[1].plot(np.arange(NUM_STEPS),vals[0],label=m_name)
        ax[1].fill_between(np.arange(NUM_STEPS),vals[1],vals[2],alpha=0.5)
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Angle alignment with target")
    ax[1].set_title("Alignment with target over time")
    ax[1].legend()

    for m_name,vals in dict_jerk.items():
        ax[2].plot(np.arange(NUM_STEPS-2),vals[0],label=m_name)
        ax[2].fill_between(np.arange(NUM_STEPS-2),vals[1],vals[2],alpha=0.3)
    ax[2].set_xlabel("Steps")
    ax[2].set_ylabel("Magnitude of jerk (m/s^3)")
    ax[2].set_title("Magnitude of jerk over time")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    use_gui = True
    main(use_gui)