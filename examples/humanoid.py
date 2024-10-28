import sys

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import MPPI, PredictiveSampling, CEM
from hydrax.mpc import run_interactive
from hydrax.tasks.humanoid import Humanoid

"""
Run an interactive simulation of the humanoid task.
"""

# Define the task (cost and dynamics)
task = Humanoid()

# Set up the controller
# Set the controller based on command-line arguments
if len(sys.argv) == 1 or sys.argv[1] == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)
elif sys.argv[1] == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=128, noise_level=0.2, temperature=0.1)
elif sys.argv[1] == "cem":
    print("Running CEM")
    ctrl = CEM(task, num_samples=128, num_elites=5, sigma_start=0.5, sigma_min=0.2)
else:
    print("Usage: python humanoid.py [ps|mppi]")
    sys.exit(1)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
start_state = np.concatenate(
    [mj_model.keyframe("stand").qpos, np.zeros(mj_model.nv)]
)

# Run the interactive simulation
run_interactive(
    mj_model,
    ctrl,
    start_state,
    frequency=30,
    show_traces=True,
    max_traces=1,
)
