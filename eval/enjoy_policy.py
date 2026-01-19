# eval/enjoy_policy.py

import sys
import os
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from motogp.env import MotoGPEnv


def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def main():
    print("Sélection du fichier de circuit (.csv)...")
    csv_path = select_file(
        "Sélectionnez le fichier CSV du circuit",
        [("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
    )

    if not csv_path:
        print("Aucun circuit sélectionné, arrêt.")
        return

    print(f"Circuit : {csv_path}")

    print("Sélection du modèle PPO (.zip)...")
    model_path = select_file(
        "Sélectionnez le modèle PPO (*.zip)",
        [("Modèle Stable-Baselines3", "*.zip"), ("Tous les fichiers", "*.*")]
    )

    if not model_path:
        # chemin par défaut
        default = os.path.join("models", "motogp_ppo.zip")
        print(f"Aucun modèle sélectionné, tentative avec : {default}")
        model_path = default

    if not os.path.isfile(model_path):
        print(f"Modèle introuvable : {model_path}")
        return

    print(f"Modèle : {model_path}")

    # Création de l'env
    env = MotoGPEnv(csv_path)

    # Chargement du modèle
    model = PPO.load(model_path, env=env)

    # Un seul épisode
    obs, info = env.reset()
    terminated = False
    truncated = False

    total_reward = 0.0

    while not (terminated or truncated):
        # action déterministe pour "montrer" la policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()  # doit afficher/mettre à jour la moto sur le circuit

    reason = info.get("terminated_reason", "unknown")
    print("=== Episode terminé ===")
    print(f"Raison : {reason}")
    print(f"Reward total : {total_reward:.2f}")
    print(f"Temps total (s) : {env.t:.2f}")
    print(f"Distance parcourue (m) : {env.s:.2f}")

    env.close()


if __name__ == "__main__":
    main()
