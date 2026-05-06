"""
Script to train the Kinova arm environment with a vectorized environment
"""
from datetime import datetime
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from gym_env_angle import ArmEnv, RewardPrintCallback

def make_env():
    """
    Wrapper function to return the env wrapped in a time limit
    """ 
    return TimeLimit(ArmEnv(), max_episode_steps=1200)

if __name__ == "__main__":
    now = datetime.now()
    time_str = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

    # Register environment in gymnasium
    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=1200,
    )

    # Check a single raw env first
    raw_env = gym.make("KinovaEnv")
    check_env(raw_env)
    raw_env.close()

    # Build SB3 vectorized env
    env = make_vec_env(
        make_env,
        n_envs=8,
        seed=0,
        vec_env_cls=SubprocVecEnv,
    )

    # Normalization ensures observations and rewards are roughly of constant scale
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Settings for PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu', # performs better on CPU than GPU as per docs + my own testing 
        policy_kwargs=dict(net_arch=[256,256]),
        learning_rate=0.00004,
        ent_coef=0.005, # how much to encourage exploration
        clip_range=0.2, # how much to allow the policy to change
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        tensorboard_log=f"../logs/ppo_kinova_{time_str}_no_curric_2x256_n8_seeds_nsteps1024_bat256"
    )

    # Which reward terms to log the contribution of in the callback
    cb_keys = ["rew_dist","rew_vel_sq","rew_align","rew_gripper","rew_progress","rew_at_target"]
    log_name = "ppo_progress_weight_n8_cr0.2_ent.005_rand1"
    
    try:
        env.env_method("set_random_level",0.5)
        model.learn(total_timesteps=16_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name)
        #model.learn(total_timesteps=2_000_000,callback=RewardPrintCallback(keys={}))
        # env.env_method("set_random_level", 0)
        # model.learn(total_timesteps=3_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.02)
        # model.learn(total_timesteps=2_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.07)
        # model.learn(total_timesteps=1_200_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.1)
        # model.learn(total_timesteps=1_200_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.15)
        # model.learn(total_timesteps=800_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.25)
        # model.learn(total_timesteps=3_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.4)
        # model.learn(total_timesteps=5_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        # env.save(f"vecnorm_{time_str}.pkl")
        # env.env_method("set_random_level", 0.55)
        # model.learn(total_timesteps=5_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #     reset_num_timesteps=False)
        #env.env_method("set_random_level", 0.5)
        #model.learn(total_timesteps=800_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
        #    reset_num_timesteps=False)
        # env.env_method("set_random_level", 0.5)
        # model.learn(total_timesteps=3_200_000,callback=RewardPrintCallback(keys=cb_keys))
    except KeyboardInterrupt:
        print("Interrupted; saving now...")
    finally:
        model.save(f"kinova_{time_str}")
        env.save(f"vecnorm_{time_str}.pkl")