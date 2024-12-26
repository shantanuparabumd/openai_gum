import mujoco
from mujoco.viewer import launch_passive

# Load the MJCF model
model = mujoco.MjModel.from_xml_path("urdf/quad_robo.xml")
data = mujoco.MjData(model)

# Launch the viewer
launch_passive(model, data)
