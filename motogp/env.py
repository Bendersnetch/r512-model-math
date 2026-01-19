# motogp/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


from .physics import (
    G,
    accel,
    lean_angle_deg,
    LEAN_MAX_DEG,
    LEAN_RATE_MAX_DEG_PER_S,
    MU_TIRE,
    GRIP_TOL,
    V_MAX_STRAIGHT,
)
from .track import precompute_track_center


class MotoGPEnv(gym.Env):
    """
    Environnement RL pour MotoGP sur un circuit donné.

    Action space : Box([0,0,-1], [1,1,1]) -> (throttle, brake, steer)
    Observation : vecteur float (7 dims) :

        0: v_norm           (v / V_MAX_STRAIGHT)
        1: d_norm           (décalage latéral / (w/2))
        2: kappa_0          (courbure au point courant)
        3: kappa_1          (courbure un peu plus loin)
        4: kappa_2          (courbure encore plus loin)
        5: v_over_profile   (v / v_profile)
        6: mu_usage_est     (estimation grip utilisé)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        dt: float = 0.02,
        max_episode_steps: int = 8000,
        max_lat_speed: float = 8.0,
        brake_decel: float = 15.0,
    ):
        super().__init__()

        # Charger circuit
        self.track = precompute_track_center(csv_path)
        self.s_track = self.track["s"]
        self.x_center = self.track["x"]
        self.y_center = self.track["y"]
        self.theta = self.track["theta"]
        self.w = self.track["w"]
        self.kappa = self.track["kappa"]
        self.v_profile = self.track["v_profile"]
        self.s_total = float(self.s_track[-1])

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.max_lat_speed = max_lat_speed  # m/s latéral max
        self.brake_decel = brake_decel

        # Actions : throttle, brake, steer
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observations (7 dims définies plus haut)
        obs_low = np.array([-np.inf] * 7, dtype=np.float32)
        obs_high = np.array([np.inf] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Etat interne
        self.v = 0.0         # vitesse m/s
        self.s = 0.0         # abscisse curviligne m
        self.d = 0.0         # offset latéral par rapport au centre (m)
        self.prev_phi_deg = 0.0
        self.t = 0.0
        self.step_count = 0

        # Pour render()
        self._fig = None
        self._ax = None
        self._bike_point = None

    # ------------------------
    # Gym API
    # ------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.v = 5.0  # légère vitesse de départ
        self.s = 0.0
        self.d = 0.0
        self.prev_phi_deg = 0.0
        self.t = 0.0
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        action = [throttle, brake, steer_cmd]
        throttle, brake in [0, 1], steer in [-1, 1]
        """

        self.step_count += 1

        # --- Action clamp ---
        action = np.array(action, dtype=np.float32)
        throttle = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        steer = float(np.clip(action[2], -1.0, 1.0))

        # --- Index sur le circuit ---
        idx = np.searchsorted(self.s_track, self.s, side="right") - 1
        idx = max(0, min(idx, len(self.s_track) - 1))

        k = self.kappa[idx]
        w_here = self.w[idx]
        v_prof = self.v_profile[idx]

        # --- Dynamique longitudinale ---
        a_motor = accel(self.v, throttle)  # >= 0
        a_brake = brake * self.brake_decel  # >= 0
        a_long = a_motor - a_brake  # peut être < 0

        # --- Déplacement latéral (commande simple) ---
        self.d += steer * self.max_lat_speed * self.dt

        # --- Accélération latérale (centripète) ---
        a_lat = self.v ** 2 * k

        # --- Mise à jour vitesse / position ---
        v_new = self.v + a_long * self.dt
        self.v = max(v_new, 0.0)

        self.s += self.v * self.dt
        self.t += self.dt

        terminated = False
        truncated = False
        info = {}
        terminated_reason = None

        # ========= CONTRAINTES / CRASH CHECK =========

        # 1) Fin de tour (objectif : 1 tour le plus vite possible)
        if self.s >= self.s_total:
            terminated = True
            terminated_reason = "lap_complete"
            self.s = self.s_total  # clamp

        # 2) Hors piste (largeur)
        half_w = 0.5 * w_here
        if abs(self.d) > half_w + 1e-6 and not terminated:
            terminated = True
            terminated_reason = "off_track"

        # 3) Angle + vitesse d'inclinaison
        phi_deg = lean_angle_deg(self.v, k)
        if phi_deg > LEAN_MAX_DEG + 1e-6 and not terminated:
            terminated = True
            terminated_reason = "lean_exceeded"
        else:
            if self.step_count > 1 and not terminated:
                dphi = abs(phi_deg - self.prev_phi_deg)
                max_step = LEAN_RATE_MAX_DEG_PER_S * self.dt
                if dphi > max_step + 1e-6:
                    terminated = True
                    terminated_reason = "roll_rate_exceeded"
        self.prev_phi_deg = phi_deg

        # 4) Grip (cercle de friction)
        a_long_abs = abs(a_long)
        a_total = np.sqrt(a_lat ** 2 + a_long_abs ** 2)
        mu_used = a_total / (MU_TIRE * G) if MU_TIRE > 0 else 0.0
        if mu_used > 1.0 + GRIP_TOL and not terminated:
            terminated = True
            terminated_reason = "grip_exceeded"

        # 5) Limite de steps -> truncated (sécurité, pas objectif)
        if self.step_count >= self.max_episode_steps and not terminated:
            truncated = True
            terminated_reason = "max_steps"

        # ========= REWARD : CONFIGURATION FONCTIONNELLE =========
        reward = 0.0

        # 1) PÉNALITÉ TEMPS (force à finir vite)
        reward -= self.dt

        # 2) MICRO-BONUS PROGRESSION (évite piège "crasher vite = ok")
        # Très petit, juste pour orienter l'exploration
        reward += 0.005 * self.v * self.dt

        # 3) BONUS PROFIL VITESSE (guidance principale)
        if not terminated and not truncated:
            if v_prof > 1.0:
                ratio = np.clip(self.v / (v_prof + 1e-6), 0.0, 2.0)
                reward += 0.02 * ratio * self.dt

        # 4) PÉNALITÉ CRASH (dissuasion forte)
        if terminated and terminated_reason in [
            "off_track",
            "lean_exceeded",
            "roll_rate_exceeded",
            "grip_exceeded",
        ]:
            reward -= 50.0

        # 5) RÉCOMPENSE TOUR COMPLET (objectif principal)
        if terminated and terminated_reason == "lap_complete":
            reward += 100.0 - self.t

        if terminated or truncated:
            info["terminated_reason"] = terminated_reason

        obs = self._get_obs(mu_used)

        return obs, reward, terminated, truncated, info

    # ------------------------
    #  Observations & render
    # ------------------------

    def _get_obs(self, mu_used_est=None):
        """Construit le vecteur d'observation."""

        idx = np.searchsorted(self.s_track, self.s, side="right") - 1
        idx = max(0, min(idx, len(self.s_track) - 1))

        k0 = self.kappa[idx]

        def safe_index(i):
            return max(0, min(i, len(self.s_track) - 1))

        idx1 = safe_index(idx + 10)
        idx2 = safe_index(idx + 20)

        k1 = self.kappa[idx1]
        k2 = self.kappa[idx2]

        w_here = self.w[idx]
        half_w = max(0.5 * w_here, 1e-3)

        d_norm = self.d / half_w
        d_norm = np.clip(d_norm, -2.0, 2.0)

        v_prof = self.v_profile[idx]
        if v_prof < 1.0:
            v_prof = 1.0
        v_over_profile = self.v / v_prof

        v_norm = self.v / V_MAX_STRAIGHT

        if mu_used_est is None:
            # estimation du grip
            a_lat = self.v ** 2 * k0
            a_total = abs(a_lat)
            mu_used_est = a_total / (MU_TIRE * G) if MU_TIRE > 0 else 0.0

        obs = np.array(
            [
                v_norm,
                d_norm,
                k0,
                k1,
                k2,
                v_over_profile,
                mu_used_est,
            ],
            dtype=np.float32,
        )

        return obs

    def render(self, mode="human"):
        """
        Render simplifié : vue du dessus avec la piste et la moto.
        (optionnel, utile pour debug)
        """
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
            self._ax.plot(self.x_center, self.y_center, "k-", linewidth=2)
            self._ax.set_aspect("equal", "box")
            self._ax.set_title("MotoGP Env (RL) - IA en action")
            self._bike_point, = self._ax.plot([], [], "ro", markersize=8)

            # Afficher tout le circuit
            margin = 50.0
            self._ax.set_xlim(self.x_center.min() - margin, self.x_center.max() + margin)
            self._ax.set_ylim(self.y_center.min() - margin, self.y_center.max() + margin)

        idx = np.searchsorted(self.s_track, self.s, side="right") - 1
        idx = max(0, min(idx, len(self.s_track) - 1))

        xc = self.x_center[idx]
        yc = self.y_center[idx]
        theta = self.theta[idx]

        nx = -np.sin(theta)
        ny = np.cos(theta)
        x = xc + self.d * nx
        y = yc + self.d * ny

        self._bike_point.set_data([x], [y])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._bike_point = None
