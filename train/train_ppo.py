# train/train_ppo.py
import sys, os
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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

    # Modèle PPO avec hyperparamètres STABLES et ÉPROUVÉS
    print("Création du modèle PPO avec configuration fonctionnelle...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,              # Très bas = stable et prudent
        n_steps=2048,                    # Standard
        batch_size=64,                   # Standard
        n_epochs=10,                     # Standard
        gamma=0.99,                      # Standard
        gae_lambda=0.95,                 # Standard
        clip_range=0.2,                  # Standard
        clip_range_vf=None,              # Pas de clipping value function
        normalize_advantage=True,        # Normalise avantages (aide stabilité)
        ent_coef=0.01,                   # Légère exploration
        vf_coef=0.5,                     # Value function coefficient
        max_grad_norm=0.5,               # Gradient clipping
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])  # Réseau séparé pi/vf
        ),
        verbose=1,
        tensorboard_log="logs/",
    )

    # Callbacks pour sauvegarder pendant l'entraînement
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Sauvegarde tous les 50k steps
        save_path="./checkpoints/",
        name_prefix="motogp_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Entraînement avec plus de timesteps
    print("=" * 60)
    print("ENTRAÎNEMENT FONCTIONNEL - Configuration éprouvée")
    print("=" * 60)
    print(f"Learning rate: 5e-5 (très stable)")
    print(f"Sauvegardes: Tous les 50k steps dans checkpoints/")
    print("=" * 60)

    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback
    )

    # Sauvegarde finale
    model.save("models/motogp_ppo_final")
    print("\n" + "=" * 60)
    print("Entraînement terminé !")
    print("Modèle final sauvegardé dans: models/motogp_ppo_final.zip")
    print("Checkpoints disponibles dans: checkpoints/")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()