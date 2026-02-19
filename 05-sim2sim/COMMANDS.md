# 05 — Sim2Sim: Comandos y explicación

## ¿Qué se hizo?

Se tomó la política entrenada en **Isaac Lab** (módulo 04) y se corrió en **MuJoCo**
sin reentrenar. Esto se llama **sim2sim**: verificar que la red neuronal funciona
en un simulador completamente diferente antes de pasar al robot real.

```
módulo 04                módulo 05               hardware
Isaac Lab (4096 envs)  →  MuJoCo (1 env)       →  Lite6 real
entrenamiento PPO          validación sim2sim       sim2real
```

---

## Archivos creados

```
05-sim2sim/
├── sim2sim.py        ← script principal
├── policy_loader.py  ← carga el .pt de RSL-RL sin necesitar isaaclab
├── mujoco_env.py     ← entorno MuJoCo que replica el módulo 04
├── lite6_full.xml    ← MJCF del Lite6 (mismo que se usa en entrenamiento)
├── meshes_mujoco/    ← meshes STL del robot
└── setup.sh          ← crea el conda env con mujoco + dependencias
```

---

## Comandos paso a paso

### 1. Activar conda y crear el entorno

```bash
source ~/miniconda3/etc/profile.d/conda.sh

cd 05-sim2sim/
bash setup.sh

conda activate sim2sim
```

> El entorno `sim2sim` es **separado** del de isaaclab.
> Solo necesita: `mujoco`, `torch`, `scipy`, `numpy`.

---

### 2. Correr el sim2sim

```bash
# Usa el checkpoint más reciente del módulo 04 automáticamente:
python sim2sim.py

# Especificar un checkpoint concreto:
python sim2sim.py --checkpoint ../04-goal-pose-l6/logs/rsl_rl/joint_position/2026-02-18_14-34-58/model_7999.pt

# Sin ventana gráfica (solo stats en terminal):
python sim2sim.py --no_render --num_episodes 30

# Visualización a 2x velocidad:
python sim2sim.py --render_speed 2.0

# Cambiar duración de episodios:
python sim2sim.py --episode_length 8.0 --num_episodes 5
```

---

## Qué hace cada archivo

### `policy_loader.py`
Carga el checkpoint `.pt` de RSL-RL **sin instalar isaaclab ni rsl_rl**.
- Extrae los pesos del actor (MLP 2×64, activación ELU) directamente del `state_dict`
- Infiere la arquitectura desde la forma de los pesos (`actor.0.weight.shape`)
- Carga el normalizador empírico (`obs_norm_state_dict`) para normalizar observaciones

### `mujoco_env.py`
Replica el entorno del módulo 04 en MuJoCo. Todos estos valores son iguales:

| Cosa                | Isaac Lab (`env_cfg.py`)                   | MuJoCo (`mujoco_env.py`)           |
|---------------------|--------------------------------------------|------------------------------------|
| Timestep            | `sim.dt = 1/60 s`                          | `model.opt.timestep = 1/60`        |
| Decimación          | `decimation = 2`                           | 2 × `mj_step` por control         |
| PD stiffness        | `ImplicitActuatorCfg(stiffness=80)`        | `actuator_gainprm = 80`            |
| PD damping          | `ImplicitActuatorCfg(damping=4)`           | `dof_damping = 4`                  |
| Escala de acciones  | `JointPositionActionCfg(scale=0.5)`        | `target = action × 0.5`           |
| Joint pos inicial   | Todo 0                                     | `qpos = 0` en reset                |
| Observaciones       | 25 valores (6+6+7+6)                       | 25 valores (6+6+7+6)               |
| Rangos del goal     | `x:[0.25,0.35] y:[-0.2,0.2] z:[0.3,0.4]` | Igual                              |

**Vector de observaciones (25 dims):**
```
[0 :6 ]  joint_pos_rel   posición de cada joint relativa al default (default=0)
[6 :12]  joint_vel        velocidad de cada joint
[12:19]  pose_command     goal del end-effector: [x, y, z, qw, qx, qy, qz]
[19:25]  last_action      acción anterior sin escalar (output crudo de la red)
```

### `sim2sim.py`
Junta todo:
1. Encuentra el checkpoint más reciente del módulo 04
2. Carga la política con `policy_loader.py`
3. Crea el entorno MuJoCo con `mujoco_env.py`
4. Verifica que las dimensiones de observaciones coincidan
5. Corre `N` episodios de 12 segundos con visualización
6. Imprime estadísticas: recompensa media y distancia final al goal

### `lite6_full.xml` + `meshes_mujoco/`
Modelo MJCF del Lite6 — el mismo que se usa en el entrenamiento de Isaac Lab.
