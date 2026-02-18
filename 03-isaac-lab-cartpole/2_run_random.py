# =============================================================================
# MODULO 3 - Script 2: Correr el entorno con acciones aleatorias
# =============================================================================
#
# Antes de entrenar, verificamos que el entorno funciona correctamente
# corriendo el loop de RL con acciones ALEATORIAS (sin agente entrenado).
#
# Esto nos permite:
#   - Ver el entorno en 3D (el polo caera y se reiniciara)
#   - Entender el loop obs -> accion -> reward -> done -> reset
#   - Debuggear el entorno sin tener que entrenar nada
#
# Correr con (desde la carpeta de IsaacLab):
#   ./isaaclab.sh -p /ruta/a/2_run_random.py --num_envs 64
#
# O si el entorno de Isaac Lab esta activado:
#   python 2_run_random.py --num_envs 64
#
# =============================================================================

import argparse

# --- 1. AppLauncher SIEMPRE primero ---
# En Isaac Lab, AppLauncher reemplaza a SimulationApp del modulo 01 y 02.
# Hace lo mismo pero agrega soporte para argumentos CLI, video, multi-GPU, etc.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Correr el Cartpole con acciones aleatorias")
parser.add_argument("--num_envs", type=int, default=64, help="Numero de entornos paralelos")

# Agrega argumentos estandar de Isaac Lab (--headless, --device, --video, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Lanzar el simulador (DEBE ocurrir antes de cualquier import de isaaclab/omni)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Imports (despues de AppLauncher) ---
import math
import torch
from collections.abc import Sequence

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


def main():
    # -------------------------------------------------------------------------
    # Crear el entorno
    # -------------------------------------------------------------------------
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = CartpoleEnv(cfg=env_cfg)

    print("\n" + "=" * 60)
    print("  ENTORNO CARTPOLE - ACCIONES ALEATORIAS")
    print("=" * 60)
    print(f"  Entornos paralelos : {env.num_envs}")
    print(f"  Espacio de accion  : {env.cfg.action_space}  (fuerza sobre el carro)")
    print(f"  Espacio de obs     : {env.cfg.observation_space}  (ang_polo, vel_polo, pos_carro, vel_carro)")
    print(f"  Device             : {env.device}")
    print(f"  Duracion episodio  : {env.cfg.episode_length_s}s")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Loop de RL
    # -------------------------------------------------------------------------
    # Este es el loop fundamental de cualquier entrenamiento de RL:
    #
    #   while True:
    #       obs = env.reset() si es necesario
    #       action = agente(obs)     <- aqui estaria la red neuronal
    #       obs, reward, done, _ = env.step(action)
    #
    # Ahora usamos acciones ALEATORIAS en lugar de una red neuronal.

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            # Reset cada 300 pasos para no quedarnos viendo el polo caido
            if step % 300 == 0:
                obs, _ = env.reset()
                print(f"[Step {step:5d}] Reset del entorno")
                print(f"           Shape de obs['policy']: {obs['policy'].shape}")
                print(f"           Ejemplo obs[env 0]: {obs['policy'][0].tolist()}")
                print()

            # Accion aleatoria: un numero por entorno, en rango [-1, 1]
            # Shape: [num_envs, 1]
            actions = torch.randn(env.num_envs, env.cfg.action_space, device=env.device)

            # Paso del entorno
            # Retorna: obs, reward, terminated (fallo), truncated (timeout), info
            obs, reward, terminated, truncated, info = env.step(actions)

            # Imprimir estadisticas cada 50 pasos
            if step % 50 == 0:
                num_terminados = terminated.sum().item()
                num_timeout    = truncated.sum().item()
                reward_media   = reward.mean().item()
                angulo_polo_0  = obs["policy"][0][0].item()  # angulo polo, entorno 0

                print(
                    f"[Step {step:5d}] "
                    f"reward_media={reward_media:+.3f} | "
                    f"terminados={num_terminados:3.0f} | "
                    f"timeouts={num_timeout:3.0f} | "
                    f"polo[0]={angulo_polo_0:+.3f}rad"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
