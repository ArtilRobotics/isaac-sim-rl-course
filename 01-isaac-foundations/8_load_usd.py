from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd

# Cargar tu escena .usd
omni.usd.get_context().open_stage("/home/marcelo/artil-projects/RL_course/02-robot-robot-onboarding/lite6_model/lite6_flat.usd")

# Esperar a que cargue completamente
from isaacsim.core.utils.stage import update_stage
update_stage()

# Loop principal
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()