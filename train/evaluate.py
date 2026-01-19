# train/evaluate.py

import time
from stable_baselines3 import PPO
from motogp.env import MotoGPEnv


def main():
    env = MotoGPEnv("data/circuit.csv")

    model = PPO.load("models/motogp_ppo", env=env)

    obs = env.reset()
    done = False
    ep_rew = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        env.render()
        time.sleep(0.01)

    env.close()
    print("Episode terminé, raison :", info.get("terminated_reason"))
    print("Reward épisode :", ep_rew)


if __name__ == "__main__":
    main()
