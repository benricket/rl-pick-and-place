import time
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from gym_env import ArmEnv

if __name__ == "__main__":
    # Register the environment
    register(
        id="KinovaEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=200,
    )

    env = gym.make("KinovaEnv").unwrapped
    print(env)
    model = PPO.load("kinova_test_angle")
    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync viewer with latest sim state
            viewer.sync()

            # optional: slow it down so it is watchable
            time.sleep(0.04)

            if terminated or truncated:
                obs, info = env.reset()