from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Sdf, UsdLux
import numpy as np
from isaacsim.core.api.objects import VisualCuboid

#Anadimos gravedad
from isaacsim.core.prims import RigidPrim


GroundPlane(prim_path="/World/GroundPlane", z_position=0)

stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

VisualCuboid(
    prim_path="/visual_cube",
    name="visual_cube",
    position=np.array([0, 0.5, 0.5]),
    size=0.3,
    color=np.array([255, 255, 0]),
)
## Anadimos Gravedad
RigidPrim("/visual_cube")



while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
