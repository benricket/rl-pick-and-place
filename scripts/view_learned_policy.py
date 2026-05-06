import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from gymnasium.envs.registration import register
import mujoco
import mujoco.viewer
from gym_env_angle import ArmEnv
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # Register the environment
    register(
        id="KinovaEnv",
        #entry_point="gym_env_angle:ArmEnv",
        entry_point="gym_env_angle:ArmEnv",
        max_episode_steps=1200,
    )
    time_str = "2026-5-4-21-35-42"
    raw_env = DummyVecEnv([lambda: Monitor(gym.make("KinovaEnv"))])
    raw_env.env_method("set_random_level", 0.5)
    vecnorm_path = Path(f"vecnorm_{time_str}.pkl") # hacky fix to deal current naming scheme
    m_filename = f"kinova_{time_str}"

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

    obs = env.reset()

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        i = 0
        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]

            # Sync viewer with latest sim state
            viewer.sync()

            # optional: slow it down so it is watchable
            time.sleep(0.01)
            print(f"{i}: reward {reward}")
            i += 1

            if done:
                obs = env.reset()