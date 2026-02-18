from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# --- 1. Piso ---
from isaacsim.core.api.objects.ground_plane import GroundPlane
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

# --- 2. Luz ---
import omni.usd
from pxr import Sdf, UsdLux
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

# --- 3. Cubo visual ---
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
VisualCuboid(
    prim_path="/visual_cube",
    name="visual_cube",
    position=np.array([0, 0.5, 0.5]),
    size=0.3,
    color=np.array([255, 255, 0]),
)

# --- 4. Física (gravedad) ---
from isaacsim.core.prims import RigidPrim
RigidPrim("/visual_cube")

# --- 5. Colisión ---
from isaacsim.core.prims import GeometryPrim
prim = GeometryPrim("/visual_cube")
prim.apply_collision_apis()

# --- 6. Transformar (mover/rotar/escalar) ---
from isaacsim.core.prims import SingleXFormPrim
from scipy.spatial.transform import Rotation

translate_offset = np.array([1.5, -0.2, 1.0])
euler_angles = [90, -90, 180]  # degrees
quat_xyzw = Rotation.from_euler("xyz", euler_angles, degrees=True).as_quat()
# Isaac Sim espera un quaternion en formato (w, x, y, z)
rotate_offset = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
scale = np.array([1, 1.5, 0.2])

cube_in_coreapi = SingleXFormPrim(prim_path="/visual_cube")
cube_in_coreapi.set_world_pose(translate_offset, rotate_offset)
cube_in_coreapi.set_local_scale(scale)

# --- Loop principal ---
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()