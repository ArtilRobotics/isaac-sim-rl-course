"""
Configuracion del entorno: Lite6 Reach (goal pose)

El robot Lite6 tiene que mover su end-effector a una pose objetivo
que se regenera aleatoriamente cada 4 segundos.

Archivos:
    env_cfg.py        <- este archivo: todo el entorno
    rsl_rl_ppo_cfg.py <- hiperparametros del agente PPO
    mdp/rewards.py    <- funciones de recompensa custom
    train.py          <- script de entrenamiento
    play.py           <- script de visualizacion
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import mdp


# =============================================================================
# ROBOT
# =============================================================================
# Nombre del link del end-effector en el USD del Lite6.
# Se usa en rewards, comandos y acciones.
EE_LINK = "uflite_gripper_link"

LITE6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="lite6_full/lite6_full.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"joint1": 0.0, "joint2": 0.0, "joint3": 0.0,
                   "joint4": 0.0, "joint5": 0.0, "joint6": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            velocity_limit_sim={"joint1": 2.175, "joint2": 2.175, "joint3": 2.175,
                                "joint4": 2.175, "joint5": 2.175, "joint6": 2.175},
            effort_limit_sim={"joint1": 40.0, "joint2": 40.0, "joint3": 40.0,
                              "joint4": 40.0, "joint5": 40.0, "joint6": 40.0},
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# =============================================================================
# ESCENA
# =============================================================================

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )
    robot: ArticulationCfg = LITE6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


# =============================================================================
# COMANDOS: pose objetivo del end-effector (se regenera cada 4s)
# =============================================================================

@configclass
class CommandsCfg:
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.35),
            pos_y=(-0.2, 0.2),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
    )


# =============================================================================
# ACCIONES: deltas de posicion de joints (control en espacio de joints)
# =============================================================================

@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=0.5,
        use_default_offset=True,
    )


# =============================================================================
# OBSERVACIONES: lo que ve la red neuronal
# =============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # posicion actual de los joints (relativa al default)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # velocidad actual de los joints
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # pose objetivo del end-effector (posicion + cuaternion = 7 valores)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        # ultima accion enviada (para suavizar movimientos)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# EVENTOS: aleatorizar posicion inicial en cada reset
# =============================================================================

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


# =============================================================================
# RECOMPENSAS
# =============================================================================

@configclass
class RewardsCfg:
    # Error de posicion L2 (penalizacion gruesa)
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK]), "command_name": "ee_pose"},
    )
    # Error de posicion con kernel tanh (recompensa fina cerca del objetivo)
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK]), "std": 0.1, "command_name": "ee_pose"},
    )
    # Error de orientacion
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK]), "command_name": "ee_pose"},
    )
    # Penalizaciones de movimiento (que no haga movimientos bruscos)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
    )


# =============================================================================
# TERMINACIONES
# =============================================================================

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# =============================================================================
# CURRICULUM: aumenta las penalizaciones de movimiento gradualmente
# (el agente primero aprende a llegar, luego a moverse suavemente)
# =============================================================================

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


# =============================================================================
# ENTORNO FINAL
# =============================================================================

@configclass
class Lite6ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene:        SceneCfg        = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    commands:     CommandsCfg     = CommandsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()
    curriculum:   CurriculumCfg   = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
