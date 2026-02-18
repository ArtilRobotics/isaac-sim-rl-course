from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import omni.usd
from isaacsim.core.utils.stage import update_stage

# Carga tu USD del Lite6
script_dir = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(script_dir, "lite6_model", "lite6_flat.usd")
omni.usd.get_context().open_stage(usd_path)
update_stage()

# Listar todos los prims para ver la estructura
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    print(prim.GetPath(), "->", prim.GetTypeName())

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()