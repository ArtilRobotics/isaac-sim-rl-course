from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects.ground_plane import GroundPlane


GroundPlane(prim_path="/World/GroundPlane", z_position=0)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
