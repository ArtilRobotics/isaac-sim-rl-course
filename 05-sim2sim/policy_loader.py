"""
Cargador de política RSL-RL para inferencia standalone.

Extrae la red del actor y el normalizador empírico del checkpoint
entrenado en Isaac Lab, SIN necesitar rsl_rl ni isaaclab instalados.

Archivos del checkpoint RSL-RL:
    model_state_dict     -> pesos de ActorCritic (actor + critic)
    obs_norm_state_dict  -> estadísticas del normalizador empírico
    optimizer_state_dict -> estado del optimizador (no necesario)
    iter                 -> iteración de entrenamiento
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ---------------------------------------------------------------------------
# Red neuronal del actor (replica la arquitectura de RSL-RL)
# ---------------------------------------------------------------------------

def build_mlp(in_size: int, hidden_dims: list[int], out_size: int) -> nn.Sequential:
    """Construye un MLP con activación ELU entre capas (igual que RSL-RL)."""
    layers: list[nn.Module] = []
    for h in hidden_dims:
        layers += [nn.Linear(in_size, h), nn.ELU()]
        in_size = h
    layers.append(nn.Linear(in_size, out_size))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Normalizador empírico
# ---------------------------------------------------------------------------

class EmpiricalNormalizer:
    """
    Replica la normalización empírica de RSL-RL.
    Durante el entrenamiento, RSL-RL computa la media y varianza
    corridas de las observaciones y las usa para normalizarlas.
    """

    def __init__(self, norm_state_dict: dict):
        self.mean: np.ndarray | None = None
        self.var:  np.ndarray | None = None

        # RSL-RL puede guardar con diferentes nombres según la versión
        for mean_key in ("mean", "running_mean"):
            if mean_key in norm_state_dict:
                self.mean = norm_state_dict[mean_key].float().numpy()
                break

        for var_key in ("var", "running_var"):
            if var_key in norm_state_dict:
                self.var = norm_state_dict[var_key].float().numpy()
                break

        if self.mean is not None:
            print(f"[Normalizer] Cargado  mean.shape={self.mean.shape}")
        else:
            print("[Normalizer] WARN: No se encontró 'mean' → sin normalización")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.mean is None or self.var is None:
            return obs
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------
# Cargador principal
# ---------------------------------------------------------------------------

class PolicyLoader:
    """
    Carga la política RSL-RL entrenada en Isaac Lab.

    Uso:
        policy = PolicyLoader("../04-goal-pose-l6/logs/.../model_7999.pt")
        action = policy.get_action(obs)   # obs: np.ndarray de forma (num_obs,)

    El actor es una MLP con activación ELU:
        obs → [Linear(num_obs, 64), ELU, Linear(64, 64), ELU, Linear(64, 6)] → action
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

        print(f"\n[PolicyLoader] ──────────────────────────────────────")
        print(f"[PolicyLoader] Cargando: {path.name}")

        ckpt = torch.load(str(path), map_location=device, weights_only=False)

        print(f"[PolicyLoader] Claves del checkpoint: {list(ckpt.keys())}")

        # ── Actor ──────────────────────────────────────────────────────────
        model_sd = ckpt["model_state_dict"]

        # Inferir arquitectura desde las dimensiones de los pesos
        # actor.0.weight: shape [hidden_1, num_obs]  → num_obs  = shape[1]
        # actor.2.weight: shape [hidden_2, hidden_1]
        # actor.4.weight: shape [num_actions, hidden_2] (2 capas ocultas)
        num_obs     = model_sd["actor.0.weight"].shape[1]
        hidden_dim1 = model_sd["actor.0.weight"].shape[0]
        hidden_dim2 = model_sd["actor.2.weight"].shape[0]
        num_actions = model_sd["actor.4.weight"].shape[0]

        print(f"[PolicyLoader] Arquitectura inferida:")
        print(f"  num_obs     = {num_obs}")
        print(f"  hidden_dims = [{hidden_dim1}, {hidden_dim2}]")
        print(f"  num_actions = {num_actions}")

        self.actor = build_mlp(num_obs, [hidden_dim1, hidden_dim2], num_actions)

        # Extraer solo los pesos del actor (prefijo "actor.")
        actor_sd = {k[len("actor."):]: v for k, v in model_sd.items()
                    if k.startswith("actor.")}
        self.actor.load_state_dict(actor_sd)
        self.actor.eval()

        # ── Normalizador empírico ──────────────────────────────────────────
        self.normalizer: EmpiricalNormalizer | None = None

        if "obs_norm_state_dict" in ckpt:
            print("[PolicyLoader] Normalizador empírico encontrado.")
            self.normalizer = EmpiricalNormalizer(ckpt["obs_norm_state_dict"])
        else:
            print("[PolicyLoader] Sin normalizador empírico en el checkpoint.")

        # ── Info ───────────────────────────────────────────────────────────
        self.num_obs     = num_obs
        self.num_actions = num_actions
        self.device      = device

        iteration = ckpt.get("iter", "?")
        print(f"[PolicyLoader] Listo ✓  (iteración {iteration})")
        print(f"[PolicyLoader] ──────────────────────────────────────\n")

    # ──────────────────────────────────────────────────────────────────────
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Inferencia de la política.

        Args:
            obs: vector de observaciones de forma (num_obs,)

        Returns:
            action: vector de acciones de forma (num_actions,)
        """
        # Normalizar observaciones (si hay normalizador)
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)

        # Inferencia en PyTorch (sin gradientes)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)   # [1, num_obs]
        with torch.no_grad():
            action_t = self.actor(obs_t)               # [1, num_actions]

        return action_t.squeeze(0).numpy()             # [num_actions]
