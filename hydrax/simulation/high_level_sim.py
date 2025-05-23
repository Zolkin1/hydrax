import os
import time
from typing import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.low_level_controllers.rl_controller import RLController
from hydrax.utils.video import VideoRecorder
from hydrax.algs.high_level_control import HighLevelControl

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(  # noqa: PLR0912, PLR0915
    planner: HighLevelControl,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    low_level_controller: RLController,
    hl_frequency: float,
    ll_frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        record_video: Whether to record a video of the simulation.
    """
    # Report the planning horizon in seconds for debugging
    # print(
    #     f"Planning with {controller.ctrl_steps} steps "
    #     f"over a {controller.plan_horizon} second horizon "
    #     f"with {controller.num_knots} knots."
    # )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / hl_frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Figure out how many sim steps to run before controlling
    control_period = 1.0 / ll_frequency
    sim_steps_per_control = int(control_period / mj_model.opt.timestep)
    sim_steps_per_control = max(sim_steps_per_control, 1)
    step_dt = sim_steps_per_control * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Controlling at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    control_steps_per_plan = sim_steps_per_replan / sim_steps_per_control
    print(
        f"{control_steps_per_plan} control steps per plan. "
    )
    # Initialize the planner
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = planner.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(planner.optimize)
    jit_interp_func = jax.jit(planner.interp_func)

    # Warm-up the planner
    print("Jitting the planner...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit planner: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Initialize the controller
    st = time.time()
    # compute_control_jit = jax.jit(partial(low_level_controller.compute_control))
    print(f"Time to jit controller: {time.time() - st:.3f} seconds")


    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    data = jnp.zeros(low_level_controller.action_size)

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(planner.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * planner.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        sim_step_count = 0

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(planner.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # simulate the system between spline replanning steps
            if sim_step_count % sim_steps_per_replan == 0:
                ## Compute a new MPC solution
                # Do a replanning step
                plan_start = time.time()
                policy_params, rollouts = jit_optimize(mjx_data, policy_params)
                plan_time = time.time() - plan_start
                # print(f"HL time to compute: {plan_time:.5f} seconds")

                # Interpolate
                # query the control spline at the sim frequency
                # (we assume the sim freq is the same as the low-level ctrl freq)
                t_curr = mj_data.time
                tq = jnp.arange(0, control_steps_per_plan) * control_period + t_curr
                tk = policy_params.tk
                knots = policy_params.mean[None, ...]
                u_hl = jit_interp_func(tq, tk, knots)[0]  # (ss, nu)
                interp_start = sim_step_count

            if sim_step_count % sim_steps_per_control == 0:
                ## Compute a new control action
                start_time = time.time()
                action, data = low_level_controller.compute_control(mj_data, u_hl[:, sim_step_count - interp_start], data)
                end_time = time.time()
                # print(f"LL time to compute: {end_time - start_time:.5f} seconds")
                mj_data.ctrl[:] = np.array(action)

            mujoco.mj_step(mj_model, mj_data)
            sim_step_count += 1

            viewer.sync()

            # # Capture frame if recording
            # if record_video and recorder.is_recording:
            #     renderer.update_scene(mj_data, viewer.cam)
            #     frame = renderer.render()
            #     recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()
