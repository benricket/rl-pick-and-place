import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from gym_env_angle import linear_schedule
from gym_env_angle import ArmEnv, RewardPrintCallback

def make_env():
    return TimeLimit(ArmEnv(), max_episode_steps=1200)

if __name__ == "__main__":

    now = datetime.now()
    time_str = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

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
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=0.00004,
        ent_coef=0.005,
        clip_range=0.2,
        n_steps=1024,        # 8 × 256 = 2048 samples/update
        batch_size=256,
        n_epochs=10,
        tensorboard_log=f"../logs/ppo_kinova_{time_str}_curriculum_smalldelta_2x256_n8_seeds_nsteps1024_bat256/",
        #n_steps=2048,
    )

    cb_keys = ["rew_dist","rew_vel_sq","rew_align","rew_gripper","rew_progress","rew_at_target"]
    log_name = "ppo_curriculum_progress_weight_n8_cr0.2_ent.005_rand0.55"
    try:
        #model.learn(total_timesteps=2_000_000,callback=RewardPrintCallback(keys={}))
        env.env_method("set_random_level", 0)
        model.learn(total_timesteps=3_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.02)
        model.learn(total_timesteps=2_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.07)
        model.learn(total_timesteps=1_200_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.1)
        model.learn(total_timesteps=1_200_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.15)
        model.learn(total_timesteps=800_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.25)
        model.learn(total_timesteps=3_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.4)
        model.learn(total_timesteps=5_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
        env.save(f"vecnorm_{time_str}.pkl")
        env.env_method("set_random_level", 0.55)
        model.learn(total_timesteps=5_000_000,callback=RewardPrintCallback(keys=cb_keys),tb_log_name=log_name,
            reset_num_timesteps=False)
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