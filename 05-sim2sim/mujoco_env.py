"""
Entorno MuJoCo para el robot Lite6 — réplica exacta de Isaac Lab.

Replica:
  • Espacio de observaciones  (25 valores = 6+6+7+6)
  • Espacio de acciones       (6 deltas de posición × 0.5)
  • Control PD                (kp=80, kd=4 — igual que ImplicitActuatorCfg)
  • Timestep & decimación     (dt=1/60 s, decimation=2 → control a 30 Hz)
  • Rangos del goal           (UniformPoseCommandCfg de env_cfg.py)

Vector de observaciones (25 dims):
  [0 : 6 ]  joint_pos_rel  = qpos - default_qpos   (default = 0 para todos)
  [6 :12 ]  joint_vel      = qvel
  [12:19 ]  pose_command   = [goal_x, goal_y, goal_z, goal_qw, goal_qx, goal_qy, goal_qz]
  [19:25 ]  last_action    = acción anterior (sin escalar)
"""

import math
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────────────────────────────────────
# Constantes — deben coincidir exactamente con env_cfg.py del módulo 04
# ──────────────────────────────────────────────────────────────────────────────

KP            = 80.0           # stiffness (ImplicitActuatorCfg)
KD            = 4.0            # damping   (ImplicitActuatorCfg)
ACTION_SCALE  = 0.5            # JointPositionActionCfg scale=0.5
SIM_DT        = 1.0 / 60.0    # Isaac Lab sim.dt
DECIMATION    = 2              # Isaac Lab decimation=2  → control a 30 Hz

# Posición inicial de los joints (todo cero, igual que LITE6_CFG.init_state)
DEFAULT_JOINT_POS = np.zeros(6, dtype=np.float64)

# Rangos del goal (CommandsCfg en env_cfg.py)
GOAL_POS_X_RANGE = (0.25, 0.35)
GOAL_POS_Y_RANGE = (-0.20, 0.20)
GOAL_POS_Z_RANGE = (0.30, 0.40)
GOAL_ROLL_RANGE  = (-math.pi / 6, math.pi / 6)
GOAL_PITCH       = math.pi / 2          # fijo en π/2
GOAL_YAW_RANGE   = (-math.pi / 9, math.pi / 9)

# Nombres de articulaciones (igual que LITE6_CFG joint_names_expr)
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]

# Nombres candidatos para el body del end-effector (se prueban en orden)
EE_BODY_CANDIDATES = [
    "uflite_gripper_link",   # nombre en Isaac Lab
    "link6",
    "Link6",
    "end_effector",
    "ee",
    "flange",
    "wrist_3_link",
]


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de geometría
# ──────────────────────────────────────────────────────────────────────────────

def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convierte ángulos de Euler (XYZ intrínseco) a cuaternión [w, x, y, z].
    Replica Isaac Lab's quat_from_euler_xyz usando la convención ZYX extrínseca
    (= XYZ intrínseca), que es lo que usa UniformPoseCommandCfg.
    """
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    q = r.as_quat()                          # scipy → [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)   # → [w, x, y, z]


# ──────────────────────────────────────────────────────────────────────────────
# Entorno principal
# ──────────────────────────────────────────────────────────────────────────────

class Lite6MuJoCoEnv:
    """
    Entorno MuJoCo para el Lite6 que replica Isaac Lab.

    Uso básico:
        env    = Lite6MuJoCoEnv("robot/scene.xml")
        obs    = env.reset()
        action = policy.get_action(obs)
        obs, reward, done = env.step(action)
        env.render()
        env.close()
    """

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self, model_path: str, render: bool = True):
        print(f"\n[MuJoCoEnv] ──────────────────────────────────────")
        print(f"[MuJoCoEnv] Cargando MJCF: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        # Ajustar timestep para que coincida con Isaac Lab
        self.model.opt.timestep = SIM_DT
        self.control_dt = SIM_DT * DECIMATION

        # Índices de joints y actuadores
        self._setup_joint_indices()

        # Sobreescribir ganancias PD del MJCF con los valores de Isaac Lab
        self._setup_pd_gains()

        # Estado interno del episodio
        self.goal_pos  = np.zeros(3, dtype=np.float32)
        self.goal_quat = np.array([1, 0, 0, 0], dtype=np.float32)   # [w,x,y,z]
        self.last_action = np.zeros(6, dtype=np.float32)

        # Visualizador pasivo de MuJoCo (se puede interactuar con el mouse)
        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        print(f"[MuJoCoEnv] Joints      : {JOINT_NAMES}")
        print(f"[MuJoCoEnv] Actuadores  : {self.model.nu} total, {len(self.actuator_ids)} del brazo")
        print(f"[MuJoCoEnv] Control dt  : {self.control_dt:.4f}s ({1/self.control_dt:.1f} Hz)")
        print(f"[MuJoCoEnv] PD gains    : kp={KP}, kd={KD}")
        print(f"[MuJoCoEnv] EE body     : '{self._ee_body_name}'")
        print(f"[MuJoCoEnv] ──────────────────────────────────────\n")

    # ── Setup indices ─────────────────────────────────────────────────────────
    def _setup_joint_indices(self):
        """Mapea nombres de joints a índices de qpos/qvel y actuadores."""
        self.qpos_ids    = []
        self.qvel_ids    = []
        self.actuator_ids = []

        for name in JOINT_NAMES:
            # Joint
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise ValueError(
                    f"Joint '{name}' no encontrado en el MJCF. "
                    f"Verifica que el modelo sea del Lite6 de mujoco_menagerie."
                )
            self.qpos_ids.append(self.model.jnt_qposadr[jnt_id])
            self.qvel_ids.append(self.model.jnt_dofadr[jnt_id])

            # Actuador — buscar por nombre del joint o por transmisión al joint
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id == -1:
                # Fallback: recorrer actuadores y ver cuál actúa sobre este joint
                for i in range(self.model.nu):
                    trntype = self.model.actuator_trntype[i]
                    if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                        if self.model.actuator_trnid[i, 0] == jnt_id:
                            act_id = i
                            break
            if act_id == -1:
                raise ValueError(
                    f"Actuador para '{name}' no encontrado. "
                    f"El MJCF debe tener actuadores de posición para cada joint."
                )
            self.actuator_ids.append(act_id)

        # End-effector body
        self._ee_body_id   = None
        self._ee_body_name = "?"
        for candidate in EE_BODY_CANDIDATES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid != -1:
                self._ee_body_id   = bid
                self._ee_body_name = candidate
                break
        if self._ee_body_id is None:
            # Último body como fallback
            self._ee_body_id   = self.model.nbody - 1
            self._ee_body_name = f"body[{self._ee_body_id}] (fallback)"

    # ── Setup PD gains ────────────────────────────────────────────────────────
    def _setup_pd_gains(self):
        """
        Sobreescribe las ganancias de los actuadores de posición con los
        valores de Isaac Lab: kp=80 (stiffness), kd=4 (damping).

        En MuJoCo, el actuador de posición computa:
            τ = kp * (q_ref - q)
        El amortiguamiento del joint añade:
            τ_damp = -kd * qd
        Esto replica exactamente ImplicitActuatorCfg(stiffness=80, damping=4).
        """
        for act_id in self.actuator_ids:
            self.model.actuator_gainprm[act_id, 0] = KP    # ganancia proporcional
            self.model.actuator_biasprm[act_id, 1] = -KP   # bias = -kp (necesario para posición)

        for qvel_id in self.qvel_ids:
            self.model.dof_damping[qvel_id] = KD           # amortiguamiento

    # ── Goal sampling ─────────────────────────────────────────────────────────
    def _sample_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """Muestrea una pose objetivo aleatoria, igual que UniformPoseCommandCfg."""
        pos = np.array([
            np.random.uniform(*GOAL_POS_X_RANGE),
            np.random.uniform(*GOAL_POS_Y_RANGE),
            np.random.uniform(*GOAL_POS_Z_RANGE),
        ], dtype=np.float32)

        roll  = np.random.uniform(*GOAL_ROLL_RANGE)
        pitch = GOAL_PITCH
        yaw   = np.random.uniform(*GOAL_YAW_RANGE)
        quat  = euler_xyz_to_quat_wxyz(roll, pitch, yaw)

        return pos, quat

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self,
              goal_pos:  np.ndarray | None = None,
              goal_quat: np.ndarray | None = None) -> np.ndarray:
        """
        Reinicia el entorno:
         - Todos los joints a posición 0 (mismo que Isaac Lab default_joint_pos)
         - Nueva pose objetivo aleatoria (o la proporcionada)
         - Retorna la observación inicial
        """
        mujoco.mj_resetData(self.model, self.data)

        # Joints a posición inicial (todo cero, igual que Isaac Lab)
        for qpos_id in self.qpos_ids:
            self.data.qpos[qpos_id] = 0.0

        mujoco.mj_forward(self.model, self.data)   # actualizar cinemática

        # Nuevo goal
        if goal_pos is not None:
            self.goal_pos  = goal_pos.astype(np.float32)
            self.goal_quat = goal_quat.astype(np.float32) if goal_quat is not None else \
                             euler_xyz_to_quat_wxyz(0, GOAL_PITCH, 0)
        else:
            self.goal_pos, self.goal_quat = self._sample_goal()

        self.last_action = np.zeros(6, dtype=np.float32)

        return self._get_obs()

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        Aplica la acción y avanza la simulación.

        La acción es el output crudo de la red (sin escalar).
        Internamente se escala × 0.5 y se suma al default (0), igual que:
            JointPositionActionCfg(scale=0.5, use_default_offset=True)

        Args:
            action: np.ndarray de forma (6,) — output de la red neuronal

        Returns:
            obs    : vector de observaciones (25,)
            reward : recompensa (distancia negativa al goal)
            done   : siempre False (tiempo controlado externamente)
        """
        # Clipping de seguridad (evitar inestabilidades)
        action = np.clip(action, -10.0, 10.0)

        # Posición objetivo = default (0) + action × 0.5
        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        # Enviar comando a los actuadores
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = target_pos[i]

        # Avanzar simulación con decimación (2 pasos por paso de control)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        # Guardar última acción (sin escalar, igual que mdp.last_action en Isaac Lab)
        self.last_action = action.astype(np.float32).copy()

        obs    = self._get_obs()
        reward = self._compute_reward()
        done   = False

        return obs, reward, done

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        Construye el vector de observaciones de 25 dimensiones.

        Replica exactamente ObservationsCfg de env_cfg.py:
          [0 :6 ]  joint_pos_rel  = qpos - 0  (default=0)
          [6 :12]  joint_vel      = qvel
          [12:19]  pose_command   = [goal_x, goal_y, goal_z, qw, qx, qy, qz]
          [19:25]  last_action    = acción anterior sin escalar
        """
        joint_pos = np.array([self.data.qpos[i] for i in self.qpos_ids], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[i] for i in self.qvel_ids], dtype=np.float32)

        joint_pos_rel = joint_pos - DEFAULT_JOINT_POS.astype(np.float32)   # default=0 → igual a qpos
        pose_command  = np.concatenate([self.goal_pos, self.goal_quat])     # 3 + 4 = 7 valores

        return np.concatenate([joint_pos_rel, joint_vel, pose_command, self.last_action])

    # ── Reward ────────────────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        """
        Recompensa simplificada: penaliza distancia euclidiana al goal.
        (No se usa para el entrenamiento, solo para estadísticas en sim2sim.)
        """
        ee_pos = self.get_ee_position()
        if ee_pos is None:
            return 0.0
        dist = float(np.linalg.norm(ee_pos - self.goal_pos))
        return -dist

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_ee_position(self) -> np.ndarray | None:
        """Posición del end-effector en el frame del mundo."""
        if self._ee_body_id is None:
            return None
        return self.data.xpos[self._ee_body_id].copy()

    def get_ee_quaternion(self) -> np.ndarray | None:
        """Cuaternión del end-effector [w, x, y, z] en el frame del mundo."""
        if self._ee_body_id is None:
            return None
        return self.data.xquat[self._ee_body_id].copy()

    def get_joint_positions(self) -> np.ndarray:
        """Posiciones actuales de los joints [joint1..joint6] en radianes."""
        return np.array([self.data.qpos[i] for i in self.qpos_ids], dtype=np.float32)

    # ── Render & Close ────────────────────────────────────────────────────────
    def render(self):
        """Actualiza el viewer de MuJoCo."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        """Cierra el viewer."""
        if self._viewer is not None:
            self._viewer.close()
