"""
Script to load a trained model and train it further (fine tuning)
"""
from datetime import datetime
from pathlib import Path
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import constant_fn
from gym_env_angle import ArmEnv, RewardPrintCallback

def make_env():
    """
    Wrapper function to return the env wrapped in a time limit
    """ 
    return TimeLimit(ArmEnv(), max_episode_steps=1200)

if __name__ == "__main__":
    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=1200,
    )

    now = datetime.now()
    time_str = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"
    time_str_old = "2026-5-4-21-35-42" # model to load

    raw_env = DummyVecEnv([lambda: Monitor(gym.make("KinovaEnv"))])
    raw_env.env_method("set_random_level", 0.5)

    vecnorm_path = Path(f"vecnorm_{time_str_old}.pkl")
    m_filename = f"kinova_{time_str_old}"
    base_env = raw_env.envs[0].unwrapped

    try:
        # If normalization constants were saved, use those
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize: {vecnorm_path}")
    except FileNotFoundError:
        env = raw_env
        print(f"No VecNormalize found for {m_filename}; using raw observations")
    
    model = PPO.load(m_filename,env=env)

    # Adjust model parameters if desired
    model.learning_rate = 4e-5
    model.lr_schedule = constant_fn(4e-5)
    model.clip_range = constant_fn(0.2)
    model.ent_coef = 0.005
    model.n_epochs = 10
    model.batch_size = 256
    model.n_steps = 1024

    # Train the model and log the provided reward function contributions
    cb_keys = ["rew_dist","rew_vel_sq","rew_align","rew_gripper","rew_progress","rew_at_target"]
    log_name = f"ppo_finetune_n8_cr0.2_ent.005_rand0.4_time{time_str}"
    try:
        # Currently set to curriculum learning --- increase task difficulty over time
        env.env_method("set_random_level", 0.2)
        model.learn(total_timesteps=1_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.25)
        model.learn(total_timesteps=1_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.3)
        model.learn(total_timesteps=1_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.35)
        model.learn(total_timesteps=1_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.4)
        model.learn(total_timesteps=1_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
    except KeyboardInterrupt:
        print("Interrupted; saving now...")
    finally:
        model.save(f"kinova_{time_str}")
        env.save(f"vecnorm_{time_str}.pkl")