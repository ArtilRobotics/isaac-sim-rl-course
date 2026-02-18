# =============================================================================
# MODULO 3 - Script 4: Visualizar la politica entrenada
# =============================================================================
#
# Carga un checkpoint guardado por 3_train_ppo.py y corre la politica
# entrenada en el simulador (con ventana grafica, sin --headless).
#
# Uso:
#   # Carga automaticamente el ultimo checkpoint guardado:
#   python3 4_play.py
#
#   # O especifica la carpeta de un run particular:
#   python3 4_play.py --run logs/cartpole_ppo/2026-02-18_10-45-24
#
#   # O el .pt directamente:
#   python3 4_play.py --checkpoint logs/cartpole_ppo/2026-02-18_10-45-24/model_150.pt
#
# =============================================================================

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualizar politica entrenada del Cartpole")
parser.add_argument("--num_envs",   type=int, default=32,   help="Entornos a visualizar")
parser.add_argument("--run",        type=str, default=None, help="Carpeta del run (logs/cartpole_ppo/<timestamp>)")
parser.add_argument("--checkpoint", type=str, default=None, help="Ruta directa al archivo .pt")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports (despues de AppLauncher) ---
import math
import time
import torch
from collections.abc import Sequence

from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)


CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


# =============================================================================
# ENTORNO Y AGENTE (misma definicion que en 3_train_ppo.py)
# =============================================================================

@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 1
    observation_space = 4
    state_space = 0
    action_scale = 100.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True)
    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]
    rew_scale_alive      =  1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos   = -1.0
    rew_scale_cart_vel   = -0.01
    rew_scale_pole_vel   = -0.005


class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(1),
            self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(1),
            self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(1),
            self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(1),
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.cfg.rew_scale_alive, self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos, self.cfg.rew_scale_cart_vel, self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]], self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]], self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape, joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]
        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float, rew_scale_terminated: float,
    rew_scale_pole_pos: float, rew_scale_cart_vel: float, rew_scale_pole_vel: float,
    pole_pos: torch.Tensor, pole_vel: torch.Tensor,
    cart_pos: torch.Tensor, cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    rew_alive       = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos    = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(1), dim=-1)
    rew_cart_vel    = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(1), dim=-1)
    rew_pole_vel    = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(1), dim=-1)
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel


@configclass
class CartpolePPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations    = 150
    save_interval     = 50
    experiment_name   = "cartpole"
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# UTILIDAD: encontrar el checkpoint mas reciente
# =============================================================================

def find_checkpoint(run_dir: str | None, checkpoint: str | None) -> str:
    """Retorna la ruta al .pt a cargar."""

    # 1. Checkpoint especificado directamente
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")
        return checkpoint

    # 2. Run especificado -> buscar el .pt con numero mas alto
    if run_dir:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"No existe la carpeta: {run_dir}")
        pts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")))
        if not pts:
            raise FileNotFoundError(f"No hay checkpoints en: {run_dir}")
        return pts[-1]  # el de mayor numero de iteracion

    # 3. Autodetectar: buscar el run mas reciente en logs/cartpole_ppo/
    log_base = os.path.join(os.path.dirname(__file__), "logs", "cartpole_ppo")
    if not os.path.isdir(log_base):
        raise FileNotFoundError(
            f"No encontre logs en '{log_base}'.\n"
            "Primero entrena con: python3 3_train_ppo.py --num_envs 2048 --headless"
        )
    runs = sorted(glob.glob(os.path.join(log_base, "*")))
    if not runs:
        raise FileNotFoundError(f"No hay runs en: {log_base}")

    # tomar el run mas reciente
    latest_run = runs[-1]
    pts = sorted(glob.glob(os.path.join(latest_run, "model_*.pt")))
    if not pts:
        raise FileNotFoundError(f"No hay checkpoints en el run: {latest_run}")
    return pts[-1]


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1. Encontrar el checkpoint
    # -------------------------------------------------------------------------
    resume_path = find_checkpoint(args_cli.run, args_cli.checkpoint)
    print(f"\n[INFO] Cargando checkpoint: {resume_path}")

    # -------------------------------------------------------------------------
    # 2. Crear el entorno (sin headless para ver la ventana)
    # -------------------------------------------------------------------------
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device

    env = CartpoleEnv(cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # 3. Cargar la politica entrenada
    # -------------------------------------------------------------------------
    agent_cfg = CartpolePPOCfg()
    agent_cfg.device = args_cli.device

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # get_inference_policy() retorna una funcion que dado obs -> accion
    policy = runner.get_inference_policy(device=env.device)

    print("\n" + "=" * 60)
    print("  PLAY - CARTPOLE ENTRENADO")
    print("=" * 60)
    print(f"  Checkpoint  : {os.path.basename(resume_path)}")
    print(f"  Entornos    : {env.num_envs}")
    print(f"  Device      : {env.device}")
    print("=" * 60 + "\n")
    print("  Cierra la ventana o presiona Ctrl+C para salir.")
    print()

    # -------------------------------------------------------------------------
    # 4. Loop de inferencia (sin gradientes, sin entrenamiento)
    # -------------------------------------------------------------------------
    obs = env_wrapped.get_observations()
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # La politica entrenada decide la accion (no aleatoria)
            actions = policy(obs)

            # Paso del entorno
            obs, rewards, dones, _ = env_wrapped.step(actions)

            if step % 100 == 0:
                print(
                    f"[Step {step:5d}] "
                    f"reward_media={rewards.mean().item():+.3f} | "
                    f"episodios_terminados={dones.sum().item():.0f}"
                )

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
