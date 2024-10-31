import mujoco

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive
from hydrax.tasks.humanoid import Humanoid

"""
Run an interactive simulation of the humanoid task.
"""

if __name__ == "__main__":
    # Define the task (cost and dynamics)
    task = Humanoid()

    # Set up the controller
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

    # Define the model used for simulation
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
    )
