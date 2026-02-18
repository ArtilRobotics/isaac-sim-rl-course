# =============================================================================
# MODULO 3 - Script 1: Anatomia del entorno DirectRL en Isaac Lab
# =============================================================================
#
# Este archivo define el entorno del Cartpole usando la API DirectRLEnv.
# NO tiene AppLauncher ni main() - es solo la DEFINICION del entorno.
# Los scripts 2 y 3 lo importan para correrlo y entrenarlo.
#
# El Cartpole:
#   - Un carro que se desliza horizontalmente  (joint: slider_to_cart)
#   - Un polo que cuelga del carro y puede rotar (joint: cart_to_pole)
#   - Objetivo: aplicar fuerzas al carro para mantener el polo vertical
#
# DirectRLEnv vs ManagerBasedRLEnv:
#   - DirectRLEnv: tu escribes TODO manualmente (obs, reward, done, reset)
#     -> Mas verboso pero mas explicito. Ideal para aprender.
#   - ManagerBasedRLEnv: usas "managers" que componen la logica desde piezas
#     -> Mas modular pero mas abstracto. Ideal para produccion.
#
# =============================================================================

from __future__ import annotations

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

# =============================================================================
# CONFIGURACION DEL ROBOT
# =============================================================================
# Esto es lo que normalmente viene en isaaclab_assets/robots/cartpole.py
# Lo definimos aqui para que quede visible y puedas adaptarlo a tu propio robot.
#
# ArticulationCfg describe COMO cargar el robot en la escena:
#   - spawn:       de donde viene el modelo (USD, URDF) y sus propiedades de fisica
#   - init_state:  posicion y orientacion inicial
#   - actuators:   que joints se pueden controlar y como
#
# Para tu propio robot solo cambiarias:
#   - usd_path:          ruta a tu USD/URDF
#   - joint_names_expr:  nombres de tus joints
#   - effort_limit_sim / stiffness / damping: propiedades de tus actuadores

CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # USD en el servidor Nucleus local que instala Isaac Sim
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
    # Posicion inicial del robot en el mundo [m] y angulos de joints [rad]
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
    ),
    # Actuadores: definen como el simulador aplica fuerzas a cada joint
    # ImplicitActuatorCfg = el motor lo maneja PhysX internamente (mas rapido)
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,  # fuerza maxima [N]
            stiffness=0.0,           # sin resorte (control por fuerza pura)
            damping=10.0,            # algo de amortiguacion para estabilidad
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,             # el polo gira libremente (sin friccion)
        ),
    },
)


# =============================================================================
# 1. CONFIGURACION DEL ENTORNO
# =============================================================================
# @configclass es como @dataclass de Python pero con utilidades extra de Isaac Lab
# (serializable a YAML, merge con replace(), etc.)

@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    """Toda la configuracion del entorno en un solo lugar."""

    # --- Parametros del episodio ---
    decimation = 2           # cada 2 pasos de simulacion -> 1 paso de RL
    episode_length_s = 5.0   # duracion maxima del episodio [segundos]

    # --- Espacios de accion y observacion ---
    # action_space = 1  ->  1 numero continuo: fuerza sobre el carro [N]
    # observation_space = 4  ->  [ang_polo, vel_polo, pos_carro, vel_carro]
    action_space = 1
    observation_space = 4
    state_space = 0           # sin estado separado (se usa obs directamente)

    action_scale = 100.0      # la red da valores en ~[-1,1], lo escalamos a N

    # --- Simulacion ---
    # dt=1/120s -> 120 Hz de fisica. Con decimation=2, el agente actua a 60 Hz.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # --- Robot ---
    # CARTPOLE_CFG ya tiene physics, actuadores, etc configurados.
    # replace() cambia el prim_path sin tocar nada mas.
    # El pattern "env_.*" le dice a Isaac Lab que hay N copias en paralelo.
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    cart_dof_name = "slider_to_cart"  # nombre del joint del carro (prismatico)
    pole_dof_name = "cart_to_pole"    # nombre del joint del polo (continuo)

    # --- Escena ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,          # cuantos entornos en paralelo (GPU los corre todos)
        env_spacing=4.0,        # distancia entre cada copia [m]
        replicate_physics=True, # reutiliza el grafo de fisica (mas rapido)
        clone_in_fabric=True,
    )

    # --- Condiciones de reset ---
    max_cart_pos = 3.0                        # limite del carro [m]
    initial_pole_angle_range = [-0.25, 0.25]  # angulo inicial del polo [rad]

    # --- Escalas de recompensa ---
    rew_scale_alive      =  1.0   # +1 por cada paso que no termina
    rew_scale_terminated = -2.0   # penalizacion al caerse
    rew_scale_pole_pos   = -1.0   # penaliza angulo del polo (quiere que sea 0)
    rew_scale_cart_vel   = -0.01  # penaliza velocidad del carro (movimiento innecesario)
    rew_scale_pole_vel   = -0.005 # penaliza velocidad angular del polo


# =============================================================================
# 2. EL ENTORNO: la clase que implementa toda la logica de RL
# =============================================================================

class CartpoleEnv(DirectRLEnv):
    """
    Entorno DirectRL del Cartpole.

    Hereda de DirectRLEnv que hereda de gymnasium.Env (interfaz estandar de RL).
    Tenemos que implementar 6 metodos:
        _setup_scene()       -> construir la escena
        _pre_physics_step()  -> preparar acciones antes de simular
        _apply_action()      -> aplicar acciones al robot
        _get_observations()  -> que ve el agente
        _get_rewards()       -> cuanta recompensa recibe
        _get_dones()         -> cuando termina el episodio
        _reset_idx()         -> como se reinicia
    """

    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Indices de los joints que nos interesan.
        # find_joints() busca por nombre y devuelve (indices, nombres).
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.action_scale = self.cfg.action_scale

        # Guardamos referencias a los tensores de estado (no copias).
        # Shape: [num_envs, num_dofs]
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        """
        Construye la escena. Se llama UNA VEZ al inicio.

        Aqui creamos el robot, el suelo, las luces, y le decimos a Isaac Lab
        que clone todo N veces (una por entorno paralelo).
        """
        # Crear el robot (Articulation es la clase para robots con joints)
        self.cartpole = Articulation(self.cfg.robot_cfg)

        # Suelo plano
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clonar: Isaac Lab copia automaticamente el robot num_envs veces
        self.scene.clone_environments(copy_from_source=False)

        # En CPU hay que filtrar colisiones entre entornos (en GPU no hace falta)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Registrar el robot en la escena para que Isaac Lab lo gestione
        self.scene.articulations["cartpole"] = self.cartpole

        # Luz de ambiente
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------------------------------
    # El loop de RL llama estos metodos en orden cada paso:
    # pre_physics -> (simular N veces) -> apply_action -> obs -> reward -> dones
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Prepara las acciones ANTES de que corra la fisica.
        Shape de actions: [num_envs, action_space] = [N, 1]
        """
        # La red neuronal da valores normalizados. Los escalamos a Newtons.
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        """
        Aplica las acciones al simulador.
        Llamado en cada sub-paso de fisica (decimation veces por paso de RL).
        """
        # set_joint_effort_target aplica fuerza/torque al joint indicado
        self.cartpole.set_joint_effort_target(
            self.actions, joint_ids=self._cart_dof_idx
        )

    def _get_observations(self) -> dict:
        """
        Que puede OBSERVAR el agente.

        Retorna un dict con key "policy" -> tensor [num_envs, observation_space].

        Nuestras 4 observaciones:
            obs[0]: angulo del polo     [rad]   (0 = vertical, pi/2 = caido)
            obs[1]: vel angular del polo [rad/s]
            obs[2]: posicion del carro  [m]
            obs[3]: velocidad del carro [m/s]
        """
        obs = torch.cat(
            [
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),  # angulo polo
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),  # vel polo
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),  # pos carro
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),  # vel carro
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Cuanta recompensa recibe cada entorno este paso.
        Retorna tensor [num_envs].

        La funcion de recompensa define QUE queremos que aprenda el agente.
        """
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cuando termina el episodio de cada entorno.

        Retorna: (terminated, time_out)
            terminated [num_envs, bool]: fallo (polo cayo, carro salio de limites)
            time_out   [num_envs, bool]: llego al tiempo maximo
        """
        # Actualizamos los tensores de estado (pueden haber cambiado en la simulacion)
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # Time out: el episodio llego al maximo de pasos
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminacion por falla: carro fuera de rango
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos,
            dim=1,
        )
        # Terminacion por falla: polo caido (angulo > 90 grados = pi/2 rad)
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2,
            dim=1,
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reinicia los entornos indicados con posiciones iniciales aleatorias.

        Isaac Lab llama esto automaticamente cuando _get_dones() devuelve True
        para alguno de los entornos.

        env_ids: lista de indices de entornos a reiniciar.
        """
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        # Posicion de partida = default + un poco de ruido en el polo
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        # Origen de cada entorno en el mundo (Isaac Lab los separa en una grilla)
        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Guardar y escribir al simulador
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# =============================================================================
# 3. FUNCION DE RECOMPENSA
# =============================================================================
# @torch.jit.script la compila a TorchScript -> se ejecuta mas rapido en GPU.
# No puede tener clases de Python ni objetos complejos, solo tensores y escalares.

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula la recompensa total.

    Componentes:
        alive:       +1 por cada paso que el polo sigue en pie
        termination: -2 si el episodio termino por falla (polo cayo)
        pole_pos:    penaliza el angulo al cuadrado (suave cerca de 0, fuerte lejos)
        cart_vel:    penaliza velocidad del carro (que se mueva lo menos posible)
        pole_vel:    penaliza velocidad angular del polo
    """
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
