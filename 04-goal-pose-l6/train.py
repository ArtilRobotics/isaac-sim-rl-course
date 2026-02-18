"""Script para entrenar el Lite6 reach con PPO y RSL-RL."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Entrenar el Lite6 reach con PPO.")
parser.add_argument("--num_envs",       type=int,  default=None,  help="Numero de entornos paralelos.")
parser.add_argument("--max_iterations", type=int,  default=None,  help="Iteraciones de entrenamiento.")
parser.add_argument("--seed",           type=int,  default=None,  help="Semilla aleatoria.")
parser.add_argument("--video",          action="store_true", default=False, help="Grabar video durante training.")
parser.add_argument("--video_length",   type=int,  default=200,   help="Duracion del video en pasos.")
parser.add_argument("--video_interval", type=int,  default=2000,  help="Cada cuantos pasos grabar video.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports (despues de AppLauncher) ---
import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from env_cfg import Lite6ReachEnvCfg
from rsl_rl_ppo_cfg import JointPositionPPORunnerCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    env_cfg   = Lite6ReachEnvCfg()
    agent_cfg = JointPositionPPORunnerCfg()

    # Overrides desde CLI
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        env_cfg.seed   = args_cli.seed
        agent_cfg.seed = args_cli.seed

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Directorio de logs
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir       = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logs en: {log_dir}")

    # Crear entorno
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video opcional
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Grabando videos durante entrenamiento.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
