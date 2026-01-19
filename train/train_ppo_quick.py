# train/train_ppo_quick.py
# Version RAPIDE pour tests (500k timesteps ~20 min)
import sys, os
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from motogp.env import MotoGPEnv


def select_csv_file():
    """
    Ouvre une boîte de dialogue Tkinter pour sélectionner un fichier CSV.
    Retourne le chemin complet.
    """
    root = tk.Tk()
    root.withdraw()

    csv_path = filedialog.askopenfilename(
        title="Sélectionnez le fichier CSV du circuit",
        filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
    )

    root.destroy()
    return csv_path


def main():
    print("Sélection du fichier de circuit (.csv)...")
    csv_path = select_csv_file()

    if not csv_path:
        print("Aucun fichier sélectionné. Arrêt.")
        return

    print(f"Fichier sélectionné : {csv_path}")

    # Construction de l'environnement RL
    env = MotoGPEnv(csv_path)

    # Modèle PPO - VERSION RAPIDE
    print("Création du modèle PPO (version rapide)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,              # Stable
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
        tensorboard_log="logs/",
    )

    # Callbacks
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/",
        name_prefix="motogp_ppo_quick",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Entraînement RAPIDE
    print("=" * 60)
    print("ENTRAÎNEMENT RAPIDE - Test de configuration")
    print("=" * 60)
    print(f"Total timesteps: 500,000 (~20 min)")
    print("=" * 60)

    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback
    )

    # Sauvegarde
    model.save("models/motogp_ppo_quick")
    print("\n" + "=" * 60)
    print("Entraînement rapide terminé !")
    print("Modèle sauvegardé dans: models/motogp_ppo_quick.zip")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
