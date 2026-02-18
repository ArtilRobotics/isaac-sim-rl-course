"""Visualizar la politica entrenada del Lite6 Reach.

Uso:
    # Carga el ultimo checkpoint automaticamente:
    python3 play.py

    # O especifica el .pt directamente:
    python3 play.py --checkpoint logs/rsl_rl/joint_position/2025-12-19_11-15-39/model_50.pt
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualizar politica entrenada del Lite6 Reach.")
parser.add_argument("--num_envs",   type=int, default=50,   help="Entornos a visualizar.")
parser.add_argument("--checkpoint", type=str, default=None, help="Ruta al archivo .pt.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports (despues de AppLauncher) ---
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from env_cfg import Lite6ReachEnvCfg
from rsl_rl_ppo_cfg import JointPositionPPORunnerCfg


def find_checkpoint(checkpoint_arg: str | None) -> str:
    """Encuentra el checkpoint mas reciente si no se especifica uno."""
    if checkpoint_arg:
        if not os.path.isfile(checkpoint_arg):
            raise FileNotFoundError(f"No existe: {checkpoint_arg}")
        return checkpoint_arg

    log_base = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", "joint_position")
    if not os.path.isdir(log_base):
        raise FileNotFoundError(
            f"No hay logs en '{log_base}'.\n"
            "Primero entrena con: python3 train.py --num_envs 4096 --headless"
        )
    runs = sorted(glob.glob(os.path.join(log_base, "*")))
    pts  = sorted(glob.glob(os.path.join(runs[-1], "model_*.pt")))
    if not pts:
        raise FileNotFoundError(f"No hay checkpoints en: {runs[-1]}")
    return pts[-1]


def main():
    resume_path = find_checkpoint(args_cli.checkpoint)
    print(f"\n[INFO] Cargando checkpoint: {resume_path}")

    # Entorno con menos envs para visualizar
    env_cfg = Lite6ReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device
    env_cfg.observations.policy.enable_corruption = False  # sin ruido en play

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=JointPositionPPORunnerCfg().clip_actions)

    agent_cfg = JointPositionPPORunnerCfg()
    agent_cfg.device = args_cli.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    print(f"  Entornos : {env_cfg.scene.num_envs}")
    print(f"  Device   : {args_cli.device}\n")
    print("  Cierra la ventana o Ctrl+C para salir.\n")

    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
