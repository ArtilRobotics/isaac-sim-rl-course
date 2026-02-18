from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# --- Imports ---
import os
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import update_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import SingleArticulation

# Habilitar importador URDF
enable_extension("isaacsim.asset.importer.urdf")
update_stage()

from isaacsim.asset.importer.urdf import _urdf

# ============================================
# 1. CREAR EL WORLD
# ============================================
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# ============================================
# 2. IMPORTAR EL ROBOT (URDF)
# ============================================
urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True

script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_dir = os.path.join(script_dir, "lite6_model")
urdf_file = "lite6_full.urdf"
urdf_path = os.path.join(urdf_dir, urdf_file)
robot = urdf_interface.parse_urdf(urdf_dir, urdf_file, import_config)
robot_prim_path = urdf_interface.import_robot(urdf_dir, urdf_file, robot, import_config)
print(f"Robot importado en: {robot_prim_path}")
update_stage()

# ============================================
# 3. CREAR EL ARTICULATION (el "controlador" del robot)
# ============================================
# Sin esto, el robot es solo un modelo 3D bonito.
# Con SingleArticulation, se convierte en algo que puedes MOVER.
arm = SingleArticulation(
    prim_path=robot_prim_path,
    name="lite6"
)

# Agregar al world y resetear (OBLIGATORIO antes de usar)
my_world.scene.add(arm)
my_world.reset()

# ============================================
# 4. INSPECCIONAR - ¿Qué tiene este robot?
# ============================================
print("\n" + "=" * 50)
print("    INFORMACIÓN DEL ROBOT LITE6")
print("=" * 50)
print(f"  Grados de libertad (DOF): {arm.num_dof}")
print(f"  Nombres de joints:        {arm.dof_names}")
print(f"  Posiciones iniciales:      {arm.get_joint_positions()}")
print(f"  Velocidades iniciales:     {arm.get_joint_velocities()}")
print("=" * 50 + "\n")

# ============================================
# 5. CICLOS DE MOVIMIENTO
# ============================================
# Vamos a hacer 4 ciclos:
#   Ciclo 0: Robot quieto (observamos posición inicial)
#   Ciclo 1: Mover a una pose
#   Ciclo 2: Volver a home (ceros)
#   Ciclo 3: Mover a otra pose + imprimir posiciones

# Definir algunas poses interesantes para el Lite6 (6 joints)
pose_home = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
pose_saludo = np.array([[-0.5, -0.3, 0.2, -1.0, 0.5, 0.3]])
pose_alcance = np.array([[0.0, -0.8, 0.3, 0.0, 1.2, 0.0]])

for i in range(4):
    print(f"\n--- CICLO {i} ---")

    if i == 0:
        print("Observando posición inicial...")

    elif i == 1:
        print("Moviendo a pose 'saludo'...")
        arm.set_joint_positions(pose_saludo)

    elif i == 2:
        print("Volviendo a HOME...")
        arm.set_joint_positions(pose_home)

    elif i == 3:
        print("Moviendo a pose 'alcance' + monitoreando...")
        arm.set_joint_positions(pose_alcance)

    # Correr 200 steps por ciclo
    for j in range(200):
        my_world.step(render=True)

        # En el último ciclo, imprimir cada 40 steps
        if i == 3 and j % 40 == 0:
            pos = arm.get_joint_positions()
            vel = arm.get_joint_velocities()
            print(f"  Step {j:3d} | Pos: {np.round(pos, 3)} | Vel: {np.round(vel, 3)}")

# ============================================
# 6. RESUMEN FINAL
# ============================================
print("\n" + "=" * 50)
print("    RESUMEN DE EJECUCIÓN")
print("=" * 50)
print(f"  Posición final: {np.round(arm.get_joint_positions(), 3)}")
print(f"  Se ejecutaron 4 ciclos x 200 steps = 800 steps")
print("=" * 50)

# Mantener ventana abierta
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()