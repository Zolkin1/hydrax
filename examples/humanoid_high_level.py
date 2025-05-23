import argparse

import jax
import jax.numpy as jnp
import mujoco
import yaml

from hydrax.algs.high_level_ps import PredictiveSamplingHL
from hydrax.low_level_controllers.g1_controller import G1Controller

# from hydrax.simulation.asynchronous import run_interactive as run_async
# from hydrax.simulation.deterministic import run_interactive
from hydrax.simulation.high_level_sim import run_interactive
from hydrax.tasks.humanoid_high_level import HumanoidHighLevel

"""
Run an interactive simulation of the humanoid standup task.
"""

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of humanoid (G1) standup."
    )
    parser.add_argument(
        "--config_file", type=str, help="Config file name in the config folder"
    )
    parser.add_argument(
        "-a",
        "--asynchronous",
        action="store_true",
        help="Use asynchronous simulation",
        default=False,
    )
    args = parser.parse_args()

    # Define the task (cost and dynamics)
    u_min = jnp.array([-1., 0., -1])
    u_max = jnp.array([1., 0., 1])
    target_pos = jnp.array([1., 0.])
    task = HumanoidHighLevel(u_max=u_max, u_min=u_min, target_pos=target_pos)

    # Parse the config file
    config_file = args.config_file
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        checkpoint_path = config["checkpoint_path"]
        dt = config["dt"]
        num_obs = config["num_obs"]
        num_action = config["num_action"]
        period = config["period"]
        robot_name = config["robot_name"]
        action_scale = config["action_scale"]
        default_angles = jnp.array(config["default_angles"])
        qvel_scale = config["qvel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        command_scale = config["command_scale"]

    rl_controller = G1Controller(
        obs_size=74,
        action_size=21,
        hidden_dims=[512, 256, 128],
        checkpoint_path="/home/zolkin/AmberLab/Project-Isaac-RL/robot-rl/robot_rl/logs/rsl_rl/g1/2025-05-22_11-49-26/model_1999.pt",
        period=period,
        cmd_scale=command_scale,
        action_scale=action_scale,
        default_angles=default_angles,
        qvel_scale=qvel_scale,
        ang_vel_scale=ang_vel_scale,
    )

    # Set up the MPC
    ctrl = PredictiveSamplingHL(
        task,
        num_inputs=3,
        controller=rl_controller,
        num_samples=128,
        noise_level=0.3,
        num_randomizations=4,
        plan_horizon=1.5, #0.6,
        spline_type="zero",
        num_knots=8 #4,
    )

    # Define the model used for simulation (stiffer contact parameters)
    mj_model = task.mj_model
    # mj_model.opt.timestep = 0.01
    # mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state so the robot falls and needs to stand back up
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    # mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

    # Run the interactive simulation
    # if args.asynchronous:
    #     print("Running asynchronous simulation")
    #
    #     # Tighten up the simulator parameters, since it's running on CPU and
    #     # therefore won't slow down the planner
    #     mj_model.opt.timestep = 0.005
    #     mj_model.opt.iterations = 100
    #     mj_model.opt.ls_iterations = 50
    #     mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    #
    #     run_async(
    #         ctrl,
    #         mj_model,
    #         mj_data,
    #     )
    # else:
    print("Running deterministic simulation")
    run_interactive(
        planner=ctrl,
        mj_model=mj_model,
        mj_data=mj_data,
        low_level_controller=rl_controller,
        hl_frequency=5,
        ll_frequency=50,
        show_traces=True,
    )
