"""
Sim2Sim: Isaac Lab (RSL-RL/PPO) → MuJoCo

Transfiere la política entrenada del módulo 04 al simulador MuJoCo.
Esto permite verificar si la política generaliza a una física diferente
sin necesitar Isaac Lab para la inferencia.

Flujo:
    1. Cargar checkpoint .pt del módulo 04
    2. Extraer la red del actor (MLP 2×64 ELU) y el normalizador empírico
    3. Crear entorno MuJoCo con el MJCF del Lite6 (lite6_full.xml)
    4. Loop: obs → política → acción → step → render

Uso:
    # Usa el checkpoint más reciente automáticamente:
    python sim2sim.py

    # O especifica un checkpoint:
    python sim2sim.py --checkpoint ../04-goal-pose-l6/logs/rsl_rl/joint_position/.../model_7999.pt

    # Sin visualización (headless):
    python sim2sim.py --no_render --num_episodes 20
"""

import argparse
import glob
import os
import time

import numpy as np

from policy_loader import PolicyLoader
from mujoco_env    import Lite6MuJoCoEnv


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(log_dir: str) -> str:
    """Encuentra el checkpoint más reciente en el directorio de logs del módulo 04."""
    runs = sorted(glob.glob(os.path.join(log_dir, "*")))
    if not runs:
        raise FileNotFoundError(
            f"No hay runs en: {log_dir}\n"
            "Primero entrena con: cd ../04-goal-pose-l6 && python train.py"
        )
    checkpoints = sorted(
        glob.glob(os.path.join(runs[-1], "model_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No hay checkpoints en: {runs[-1]}")
    return checkpoints[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sim2Sim: transfiere política de Isaac Lab a MuJoCo."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Ruta al archivo .pt. Si no se especifica, usa el más reciente del módulo 04.",
    )
    parser.add_argument(
        "--robot_model", type=str, default="robot/scene.xml",
        help="Ruta al MJCF del robot.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Número de episodios a ejecutar.",
    )
    parser.add_argument(
        "--episode_length", type=float, default=12.0,
        help="Duración de cada episodio en segundos.",
    )
    parser.add_argument(
        "--render_speed", type=float, default=1.0,
        help="Multiplicador de velocidad de visualización (1.0 = tiempo real, 2.0 = doble velocidad).",
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="Deshabilitar visualización (modo headless).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Encontrar checkpoint ───────────────────────────────────────────────────
    if args.checkpoint is None:
        log_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "04-goal-pose-l6", "logs", "rsl_rl", "joint_position",
        )
        log_dir = os.path.normpath(log_dir)
        args.checkpoint = find_latest_checkpoint(log_dir)

    print(f"\n{'='*62}")
    print(f"  SIM2SIM: Isaac Lab (RSL-RL/PPO) → MuJoCo")
    print(f"{'='*62}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Modelo MJCF  : {args.robot_model}")
    print(f"  Episodios    : {args.num_episodes}")
    print(f"  Duración ep. : {args.episode_length}s")
    print(f"  Render       : {'no' if args.no_render else 'sí'}")
    print(f"{'='*62}")

    # ── Cargar política ───────────────────────────────────────────────────────
    policy = PolicyLoader(args.checkpoint)

    # ── Crear entorno MuJoCo ──────────────────────────────────────────────────
    env = Lite6MuJoCoEnv(
        model_path=args.robot_model,
        render=not args.no_render,
    )

    # Verificar compatibilidad de dimensiones
    obs_test = env.reset()
    if obs_test.shape[0] != policy.num_obs:
        print(f"\n⚠️  ADVERTENCIA: dimensión de observaciones no coincide!")
        print(f"   Entorno produce : {obs_test.shape[0]} valores")
        print(f"   Política espera : {policy.num_obs} valores")
        print(f"   Revisa _get_obs() en mujoco_env.py\n")
    else:
        print(f"\n✓  Dimensiones OK: obs={policy.num_obs}, actions={policy.num_actions}\n")

    # ── Loop de episodios ─────────────────────────────────────────────────────
    max_steps    = int(args.episode_length / env.control_dt)
    sleep_time   = env.control_dt / args.render_speed if not args.no_render else 0.0

    ep_rewards     = []
    ep_final_dists = []

    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0.0

        print(f"\nEpisodio {episode + 1}/{args.num_episodes}")
        print(f"  Goal pos  : x={env.goal_pos[0]:.3f}  y={env.goal_pos[1]:.3f}  z={env.goal_pos[2]:.3f}")
        print(f"  Goal quat : w={env.goal_quat[0]:.3f}  x={env.goal_quat[1]:.3f}  "
              f"y={env.goal_quat[2]:.3f}  z={env.goal_quat[3]:.3f}")

        for step in range(max_steps):
            # Inferencia de la política
            action = policy.get_action(obs)

            # Paso en el entorno
            obs, reward, _ = env.step(action)
            episode_reward += reward

            # Render
            if not args.no_render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Log cada segundo
            if (step + 1) % int(1.0 / env.control_dt) == 0:
                ee_pos  = env.get_ee_position()
                dist    = float(np.linalg.norm(ee_pos - env.goal_pos)) if ee_pos is not None else float("nan")
                elapsed = (step + 1) * env.control_dt
                print(f"  t={elapsed:.1f}s  |  dist_goal={dist:.4f}m  |  "
                      f"joints={np.round(env.get_joint_positions(), 3)}")

        # Estadísticas del episodio
        ee_pos_final = env.get_ee_position()
        final_dist   = float(np.linalg.norm(ee_pos_final - env.goal_pos)) \
                       if ee_pos_final is not None else float("nan")

        ep_rewards.append(episode_reward)
        ep_final_dists.append(final_dist)

        print(f"\n  ── Fin episodio {episode + 1} ──")
        print(f"  Recompensa acumulada : {episode_reward:.2f}")
        print(f"  Distancia final al goal : {final_dist:.4f} m")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  RESUMEN ({args.num_episodes} episodios)")
    print(f"{'='*62}")
    print(f"  Recompensa media  : {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"  Dist. final media : {np.nanmean(ep_final_dists):.4f} m ± {np.nanstd(ep_final_dists):.4f} m")
    print(f"  Mejor episodio    : {max(ep_rewards):.2f}")
    print(f"  Peor episodio     : {min(ep_rewards):.2f}")
    print(f"{'='*62}\n")

    env.close()


if __name__ == "__main__":
    main()
