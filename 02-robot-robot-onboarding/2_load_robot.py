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


# Mantener ventana abierta
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()