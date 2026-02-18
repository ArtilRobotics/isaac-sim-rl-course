# =============================================================================
# MODULO 3 - Script 3: Entrenamiento del Cartpole con PPO
# =============================================================================
#
# Aqui entrenamos el agente usando PPO (Proximal Policy Optimization),
# el algoritmo de RL mas usado en robotica con Isaac Lab.
#
# Usamos rsl_rl: la libreria de RL del grupo RSL (ETH Zurich), usada en
# ANYmal, unitree, etc. Es rapida y simple de usar con Isaac Lab.
#
# Arquitectura:
#   Entorno (N paralelos) -> rsl_rl VecEnvWrapper -> OnPolicyRunner (PPO)
#
# Los logs se guardan en ./logs/cartpole_ppo/<timestamp>/
# Se pueden visualizar con TensorBoard:
#   tensorboard --logdir logs/cartpole_ppo
#
# Correr con:
#   ./isaaclab.sh -p 3_train_ppo.py --num_envs 2048
#   ./isaaclab.sh -p 3_train_ppo.py --num_envs 2048 --max_iterations 300
#   ./isaaclab.sh -p 3_train_ppo.py --num_envs 2048 --headless  (sin ventana, mas rapido)
#
# =============================================================================

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Entrenar Cartpole con PPO")
parser.add_argument("--num_envs",        type=int, default=2048, help="Entornos paralelos")
parser.add_argument("--max_iterations",  type=int, default=150,  help="Iteraciones de PPO")
parser.add_argument("--seed",            type=int, default=42,   help="Semilla aleatoria")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports (despues de AppLauncher) ---
import math
import os
import torch
from collections.abc import Sequence
from datetime import datetime

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
# DEFINICION DEL ENTORNO (ver 1_cartpole_env.py para comentarios detallados)
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


# =============================================================================
# CONFIGURACION DEL AGENTE PPO
# =============================================================================

@configclass
class CartpolePPOCfg(RslRlOnPolicyRunnerCfg):
    """
    Hiperparametros del entrenamiento PPO para el Cartpole.

    PPO es un algoritmo on-policy: recolecta experiencia con la politica actual,
    hace varios pasos de gradiente, y descarta esa experiencia.
    """

    # --- Loop de entrenamiento ---
    num_steps_per_env = 16    # pasos por entorno antes de actualizar la politica
                               # total de experiencia por iteracion = num_envs * num_steps
    max_iterations    = 150    # iteraciones totales de PPO (~300s de entorno por iteracion)
    save_interval     = 50     # cada cuantas iteraciones guardar checkpoint

    experiment_name = "cartpole"

    # --- Red neuronal (Actor-Critic) ---
    # Actor: decide las acciones a partir de las observaciones
    # Critic: estima el valor del estado (para calcular ventajas)
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[32, 32],   # 2 capas de 32 neuronas -> suficiente para cartpole
        critic_hidden_dims=[32, 32],
        activation="elu",
        init_noise_std=1.0,           # exploracion inicial
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    # --- Hiperparametros de PPO ---
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # peso de la perdida del critic
        use_clipped_value_loss=True,  # estabilidad del critic
        clip_param=0.2,               # epsilon de PPO (cuanto puede cambiar la politica)
        entropy_coef=0.005,           # incentiva exploracion
        num_learning_epochs=5,        # pasadas sobre los datos recolectados
        num_mini_batches=4,           # mini-batches por epoch
        learning_rate=1e-3,           # tasa de aprendizaje de Adam
        schedule="adaptive",          # ajusta lr segun KL divergence
        gamma=0.99,                   # factor de descuento (horizonte de recompensa)
        lam=0.95,                     # lambda para GAE (balance bias-varianza)
        desired_kl=0.01,              # KL objetivo para schedule adaptivo
        max_grad_norm=1.0,            # clipping de gradientes
    )


# =============================================================================
# MAIN: crear entorno, configurar PPO, entrenar
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1. Crear el entorno
    # -------------------------------------------------------------------------
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device
    env_cfg.seed           = args_cli.seed

    env = CartpoleEnv(cfg=env_cfg)

    # -------------------------------------------------------------------------
    # 2. Configuracion del agente
    # -------------------------------------------------------------------------
    agent_cfg = CartpolePPOCfg()
    agent_cfg.device         = args_cli.device
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed           = args_cli.seed

    # -------------------------------------------------------------------------
    # 3. Directorio de logs
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir   = os.path.join("logs", "cartpole_ppo", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO PPO - CARTPOLE")
    print("=" * 60)
    print(f"  Entornos paralelos   : {env_cfg.scene.num_envs}")
    print(f"  Pasos por iteracion  : {env_cfg.scene.num_envs * agent_cfg.num_steps_per_env:,}")
    print(f"  Iteraciones PPO      : {agent_cfg.max_iterations}")
    print(f"  Device               : {agent_cfg.device}")
    print(f"  Logs en              : {log_dir}")
    print("=" * 60)
    print()
    print("  Para ver el progreso en tiempo real:")
    print(f"  tensorboard --logdir {os.path.abspath(log_dir)}")
    print()

    # -------------------------------------------------------------------------
    # 4. Wrap del entorno para rsl_rl
    # -------------------------------------------------------------------------
    # RslRlVecEnvWrapper adapta nuestro DirectRLEnv a la interfaz que espera
    # rsl_rl (retorna tensores en lugar de dicts, etc.)
    env_wrapped = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # 5. Crear el runner de PPO
    # -------------------------------------------------------------------------
    # OnPolicyRunner gestiona:
    #   - La red neuronal (Actor-Critic)
    #   - El buffer de experiencia
    #   - Los pasos de gradiente
    #   - El logging a TensorBoard
    #   - El guardado de checkpoints
    runner = OnPolicyRunner(
        env_wrapped,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # -------------------------------------------------------------------------
    # 6. Entrenar
    # -------------------------------------------------------------------------
    # learn() ejecuta el loop:
    #   for iter in range(max_iterations):
    #       recolectar num_steps_per_env pasos de experiencia
    #       calcular ventajas (GAE)
    #       num_learning_epochs * num_mini_batches pasos de gradiente
    #       loggear metricas
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,  # iniciar episodios en distintos puntos
    )

    print(f"\n[INFO] Entrenamiento finalizado. Modelo guardado en: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
