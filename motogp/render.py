# motogp/render.py

import time
from .env import MotoGPEnv


def rollout_random(csv_path: str, steps: int = 2000, sleep: float = 0.01):
    """
    Lance un épisode avec des actions aléatoires pour tester l'environnement.
    """
    env = MotoGPEnv(csv_path)
    obs = env.reset()
    done = False
    ep_rew = 0.0

    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        env.render()
        time.sleep(sleep)
        if done:
            print("Terminé, raison :", info.get("terminated_reason"))
            break

    env.close()
    print("Reward épisode :", ep_rew)
