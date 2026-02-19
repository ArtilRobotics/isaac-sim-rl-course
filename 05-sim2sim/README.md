# 05 — Sim2Sim: Isaac Lab → MuJoCo

Transfiere la política entrenada en el módulo 04 al simulador **MuJoCo**,
demostrando que la política funciona en una física completamente diferente.

---

## ¿Qué es Sim2Sim?

En robótica con RL, entrenamos en un simulador (Isaac Lab) y luego queremos
desplegar en el mundo real. Una forma de validar antes del hardware es hacer
**sim2sim**: transferir la política a *otro* simulador con características
físicas distintas.

```
Isaac Lab (entrenamiento)   →   MuJoCo (validación sim2sim)   →   Robot real
       PPO/RSL-RL                    este módulo                   (sim2real)
```

---

## Estructura

```
05-sim2sim/
├── sim2sim.py        ← script principal: carga política y corre en MuJoCo
├── policy_loader.py  ← extrae red del actor del checkpoint RSL-RL (sin isaaclab)
├── mujoco_env.py     ← entorno MuJoCo que replica el módulo 04 exactamente
├── get_robot.py      ← descarga el MJCF del Lite6 de mujoco_menagerie
├── setup.sh          ← crea el entorno conda con mujoco + dependencias
└── robot/            ← aquí se descarga el MJCF (auto-generado por get_robot.py)
```

---

## Setup (entorno separado de isaaclab)

```bash
# 1. Activar conda (si no está activo)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Crear entorno e instalar mujoco
bash setup.sh

# 3. Activar el entorno sim2sim
conda activate sim2sim

# 4. Descargar el modelo MJCF del Lite6 (de mujoco_menagerie)
python get_robot.py
```

---

## Ejecutar

```bash
conda activate sim2sim
cd 05-sim2sim/

# Usa el checkpoint más reciente del módulo 04 automáticamente:
python sim2sim.py

# O especifica un checkpoint concreto:
python sim2sim.py --checkpoint ../04-goal-pose-l6/logs/rsl_rl/joint_position/2026-02-18_14-34-58/model_7999.pt

# Sin visualización (más rápido para stats):
python sim2sim.py --no_render --num_episodes 30

# Control de velocidad de visualización (2x = doble velocidad):
python sim2sim.py --render_speed 2.0
```

---

## Detalles técnicos: qué se replica exactamente

| Parámetro               | Isaac Lab (`env_cfg.py`)                    | MuJoCo (`mujoco_env.py`)          |
|-------------------------|---------------------------------------------|-----------------------------------|
| Timestep                | `sim.dt = 1/60 s`                           | `model.opt.timestep = 1/60`       |
| Decimación              | `decimation = 2`                            | 2 pasos de `mj_step` por control  |
| Control dt              | `2/60 ≈ 0.033 s` (30 Hz)                   | `SIM_DT × DECIMATION`             |
| PD stiffness            | `ImplicitActuatorCfg(stiffness=80)`         | `actuator_gainprm[i,0] = 80`      |
| PD damping              | `ImplicitActuatorCfg(damping=4)`            | `dof_damping[i] = 4`              |
| Escala de acciones      | `JointPositionActionCfg(scale=0.5)`         | `target = action × 0.5`          |
| Default joint pos       | Todo 0                                      | `qpos[joint_ids] = 0` en reset    |
| Obs: joint_pos          | `joint_pos_rel = qpos - default (= 0)`      | `qpos[joint_ids]`                 |
| Obs: joint_vel          | `joint_vel_rel = qvel - default (= 0)`      | `qvel[joint_ids]`                 |
| Obs: pose_command       | `[pos_x, pos_y, pos_z, qw, qx, qy, qz]`   | Igual, muestreado aleatoriamente  |
| Obs: last_action        | Output crudo de la red (sin escalar)        | Guardado antes del escalado       |
| Total observaciones     | **25** (6+6+7+6)                            | **25**                            |

### Normalización empírica
El checkpoint RSL-RL con `empirical_normalization=True` guarda las estadísticas
de las observaciones en `obs_norm_state_dict`. `policy_loader.py` las carga y
las aplica antes de pasar las observaciones a la red.

---

## ¿Por qué podría no funcionar perfectamente?

El sim2sim no es perfecto. Las diferencias físicas entre simuladores incluyen:

1. **Modelo de colisiones** — MuJoCo usa primitivas geométricas, Isaac Lab usa
   mallas completas.
2. **Solver** — cada simulador tiene su propio solver numérico.
3. **Meshes del robot** — el MJCF de mujoco_menagerie puede diferir ligeramente
   del USD usado en entrenamiento (masas, inertias, longitudes de links).
4. **Actuadores** — aunque replicamos kp/kd, el modelo interno del actuador
   puede diferir en latencia y saturación.

Estas diferencias son exactamente lo que hace interesante el sim2sim: si la
política funciona a pesar de ellas, es más probable que funcione en hardware real.
