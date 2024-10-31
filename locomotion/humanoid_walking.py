import sys

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import MPPI, PredictiveSampling, CEM
from hydrax.mpc import run_interactive
from hydrax.tasks.humanoid_walking import HumanoidWalking
from hydrax.read_settings import read_settings

"""
Run an interactive simulation of the humanoid task.
"""

# Define the task (cost and dynamics)
settings = read_settings("locomotion/settings.yaml")
task = HumanoidWalking(settings.planning_horizon, settings.sim_steps_per_control, settings.target_vel, settings.weights, settings.gait)

# Set up the controller
# Set the controller based on command-line arguments
if len(sys.argv) == 1 or sys.argv[1] == "ps":
    print("---- Running predictive sampling ----")
    print("Num samples: " + str(settings.num_samples) + "\nNoise level: " + str(settings.noise_level))
    ctrl = PredictiveSampling(task, num_samples=settings.num_samples, noise_level=settings.noise_level)
elif sys.argv[1] == "mppi":
    print("---- Running MPPI ----")
    print("Num samples: " + str(settings.num_samples) + "\nNoise level: " + str(settings.noise_level) + "\nTemperature: " + str(settings.temperature))
    ctrl = MPPI(task, num_samples=settings.num_samples, noise_level=settings.noise_level, temperature=settings.temperature)
elif sys.argv[1] == "cem":
    print("---- Running CEM ----")
    print("Num samples: " + str(settings.num_samples) + "\nNoise level: " + str(settings.noise_level) + "\nSigma start: " + str(settings.sigma_start)
          + "\nSigma min: " + str(settings.sigma_min) + "\nNum elites: " + str(settings.num_elite))
    ctrl = CEM(task, num_samples=settings.num_samples, num_elites=settings.num_elite, sigma_start=settings.sigma_start, sigma_min=settings.sigma_min)
else:
    print("Usage: python humanoid.py [ps|mppi]")
    sys.exit(1)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/basic_scene_verify_sim.xml")

# TODO: Go back to this model when I know why its slower
# mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/g1_small_minimal_contacts.xml")
start_state = np.concatenate(
    [mj_model.keyframe("stand").qpos, np.zeros(mj_model.nv)]
)

# Run the interactive simulation
run_interactive(
    mj_model,
    ctrl,
    start_state,
    frequency=settings.ctrl_freq,
    show_traces=True,
    max_traces=3,
)
