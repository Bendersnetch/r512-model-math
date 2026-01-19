# train/rollout_random.py

import sys
import os
# Ajouter le répertoire parent au path pour importer motogp
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from motogp.env import MotoGPEnv


def main():
    env = MotoGPEnv("data/circuit.csv")
    obs, info = env.reset()
    done = False
    ep_rew = 0.0

    for _ in range(3000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward
        env.render()
        time.sleep(0.01)

        done = terminated or truncated
        if done:
            print("Episode terminé, raison :", info.get("terminated_reason"))
            break

    env.close()
    print("Reward épisode :", ep_rew)


if __name__ == "__main__":
    main()
