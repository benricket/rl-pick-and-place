"""
Script to load models, test then for a number of trials, and plot their performance
"""
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from mujoco_example import get_joint_velocities

def main():
    """
    Script for evaluating the performance of a learned policy
    """
        
    NUM_STEPS = 2000
    NUM_TRIALS = 10
    RAND_LEVEL = 0.5

    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=NUM_STEPS+1,
    )

    # Titles of models to learn and their file name
    ##### Fill this with the names of models to display + path to their zip file
    learned_models = {
        "n=1, 3x512 size": "kinova_2026-5-1-16-13-25",
        #"n=8, 2x256 size, reward norm, 0.15 rand": "kinova_2026-5-4-21-35-42",
        #"n=8, 2x256 size, 0.15 rand": "kinova_2026-5-4-20-47-26",
        "n=8, 2x256 size, reward norm, to 0.55 rand": "kinova_2026-5-5-10-2-5",
        "n=8, 2x256 size, reward norm, to 0.15 rand, no curriculum": "kinova_2026-5-5-22-46-11",
        "n=8, 2x256 size, reward norm, to 0.5 rand, no curriculum": "kinova_2026-5-5-22-44-55"}

    # Define dicts to hold logged values in
    dict_dist = {}
    dict_align = {}
    dict_jerk = {}

    for m_name,m_filename in learned_models.items():

        raw_env = DummyVecEnv([lambda: Monitor(gym.make("KinovaEnv"))])
        raw_env.env_method("set_random_level", RAND_LEVEL)

        vecnorm_path = Path(f"vecnorm_{m_filename[7:]}.pkl") # hacky fix to deal current naming scheme

        try:
            # Do we have normalization parameters? If so, load them
            env = VecNormalize.load(str(vecnorm_path), raw_env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded VecNormalize: {vecnorm_path}")
        except FileNotFoundError:
            env = raw_env
            print(f"No VecNormalize found for {m_filename}; using raw observations")

        base_env = raw_env.envs[0].unwrapped
        model = PPO.load(m_filename)

        # arrays to hold values for logging 
        arr_dist = np.zeros((NUM_TRIALS,NUM_STEPS))
        arr_align = np.zeros((NUM_TRIALS,NUM_STEPS))
        arr_jerk = np.zeros((NUM_TRIALS,NUM_STEPS - 2))

        for i in range(NUM_TRIALS):
            print(f"starting trial {i}")
            obs = env.reset()
            arr_vel = np.zeros((6,NUM_STEPS))
            for j in range(NUM_STEPS):
                action, _states = model.predict(obs, deterministic=False)
                obs, _, dones, infos = env.step(action)
                done = dones[0]
                _ = infos[0]

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
        
        # Mean and percentiles of metrics
        arr_dist_mean = np.mean(arr_dist,axis=0)
        arr_dist_p10 = np.quantile(arr_dist,0.25,axis=0)
        arr_dist_p90 = np.quantile(arr_dist,0.75,axis=0)
        arr_jerk_mean = np.mean(arr_jerk,axis=0)
        arr_jerk_p10 = np.quantile(arr_jerk,0.25,axis=0)
        arr_jerk_p90 = np.quantile(arr_jerk,0.75,axis=0)
        arr_align_mean = np.mean(arr_align,axis=0)
        arr_align_p10 = np.quantile(arr_align,0.25,axis=0)
        arr_align_p90 = np.quantile(arr_align,0.75,axis=0)

        dict_dist[m_name] = (arr_dist_mean,arr_dist_p10,arr_dist_p90)
        dict_jerk[m_name] = (arr_jerk_mean,arr_jerk_p10,arr_jerk_p90)
        dict_align[m_name] = (arr_align_mean,arr_align_p10,arr_align_p90)

    # Plot metrics on subplots
    fig,ax = plt.subplots(3,1)
    for m_name,vals in dict_dist.items():
        ax[0].plot(np.arange(NUM_STEPS),vals[0],label=m_name)
        ax[0].fill_between(np.arange(NUM_STEPS),vals[1],vals[2],alpha=0.25)
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Distance from target (m)")
    ax[0].set_title("Distance from target over time")
    ax[0].hlines(0.05,0,NUM_STEPS,colors='k',linestyle="dashed")
    ax[0].legend(fontsize='8',bbox_to_anchor=(1.02, 1), loc='upper left')

    for m_name,vals in dict_align.items():
        ax[1].plot(np.arange(NUM_STEPS),vals[0],label=m_name)
        ax[1].fill_between(np.arange(NUM_STEPS),vals[1],vals[2],alpha=0.25)
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Angle alignment with target")
    ax[1].set_title("Alignment with target over time")
    ax[1].legend(fontsize='8',bbox_to_anchor=(1.02, 1), loc='upper left')

    for m_name,vals in dict_jerk.items():
        ax[2].plot(np.arange(NUM_STEPS-2),vals[0],label=m_name)
        ax[2].fill_between(np.arange(NUM_STEPS-2),vals[1],vals[2],alpha=0.25)
    ax[2].set_xlabel("Steps")
    ax[2].set_ylabel("Magnitude of jerk (m/timestep^3)")
    ax[2].set_title("Magnitude of jerk over time")
    ax[2].legend(fontsize='8',bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.suptitle(f"Mean and 25th-75th percentile results for {RAND_LEVEL} randomness")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()