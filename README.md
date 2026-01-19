# Projet MotoGP - Optimisation de Trajectoire par Apprentissage par Renforcement

Projet de modélisation mathématique pour optimiser la trajectoire d'une moto de course sur un circuit donné.

## Prérequis

- Python 3.9 ou supérieur
- pip 

## Installation

### 1. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l'environnement virtuel

**Windows (CMD):**
```bash
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

ou manuellement :

```bash
pip install gymnasium numpy pandas matplotlib stable-baselines3 tensorboard
```


## Utilisation

### Simulation heuristique (sans IA)

Lance une simulation avec contrôleur proportionnel et racing line géométrique :

```bash
python sim.py
```

- Sélectionne le fichier circuit (data/circuit.csv)
- Choisis la vitesse d'animation (x1, x2, x5, x10)
- Observe l'animation avec télémétrie en temps réel

### Entraîner un nouvel agent IA

Entraîne un agent PPO sur le circuit :

```bash
python train/train_ppo.py
```

- Sélectionne le fichier circuit (data/circuit.csv)
- L'entraînement dure environ 5-10 minutes (500,000 timesteps)
- Le modèle est sauvegardé dans models/motogp_ppo.zip

Pour visualiser l'entraînement en temps réel avec TensorBoard (dans un terminal séparé) :

```bash
tensorboard --logdir=logs
```

Puis ouvre http://localhost:6006 dans son navigateur.

### Évaluer l'agent IA entraîné

Visualise le comportement de l'agent entraîné :

```bash
python eval/enjoy_policy.py
```

- Sélectionne le fichier circuit (data/circuit.csv)
- Sélectionne le modèle (models/motogp_ppo.zip)
- Observe l'IA piloter la moto sur le circuit
- Les performances sont affichées dans le terminal à la fin

## Paramètres physiques

Le modèle MotoGP implémenté utilise les paramètres suivants :

- Masse totale : 240 kg (moto + pilote)
- Puissance moteur : 220 kW
- Coefficient aérodynamique : 0.18
- Accélération maximale : 12 m/s² (1.2 g)
- Freinage maximal : 15 m/s² (1.5 g)
- Angle d'inclinaison maximal : 67°
- Vitesse de basculement maximale : 120 °/s
- Coefficient de friction : 1.6
- Vitesse maximale en ligne droite : 340 km/h

## Environnement d'apprentissage

### Espace d'actions

L'agent contrôle 3 paramètres continus :

1. Accélérateur : [0, 1] (0 = coupé, 1 = plein gaz)
2. Frein : [0, 1] (0 = pas de frein, 1 = freinage maximal)
3. Direction latérale : [-1, 1] (déplacement gauche/droite)

### Espace d'observations

L'agent reçoit un vecteur de 7 dimensions :

1. Vitesse normalisée
2. Position latérale normalisée
3. Courbure actuelle
4. Courbure à l'horizon 1
5. Courbure à l'horizon 2
6. Ratio vitesse actuelle / vitesse cible
7. Utilisation du grip

### Fonction de récompense

- Pénalité de temps : -0.02 par pas de temps
- Bonus de vitesse : +0.02 si proche de la vitesse optimale
- Pénalité de crash : -50 (hors piste, angle excessif, grip dépassé)
- Bonus de complétion : +100 - temps_total si tour complété

## Dépendances

- **gymnasium** : Framework d'environnements RL
- **stable-baselines3** : Implémentation de PPO
- **numpy** : Calcul numérique
- **pandas** : Manipulation des données de circuit
- **matplotlib** : Visualisation
- **tensorboard** : Suivi de l'entraînement (optionnel)