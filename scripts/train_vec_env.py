import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from gym_env_angle import linear_schedule
from gym_env_angle import ArmEnv

if __name__ == "__main__":

    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=400,
    )

    # Check a single raw env first
    raw_env = gym.make("KinovaEnv")
    check_env(raw_env)
    raw_env.close()

    # Build SB3 vectorized env
    env = make_vec_env(lambda: ArmEnv(), n_envs=8)

    # Optional but often helpful for stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(0.003, 0.002),
        tensorboard_log="../logs/ppo_kinova/",
    )

    try:
        model.learn(
            total_timesteps=500_000,
            tb_log_name="kinova_vec",
        )
    except KeyboardInterrupt:
        print("Interrupted; saving now...")
    finally:
        model.save("kinova_test_angle")
        env.save("kinova_test_angle_vecnormalize.pkl")