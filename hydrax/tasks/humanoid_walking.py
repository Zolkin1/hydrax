import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.read_settings import Weights, GaitSettings

from gait_generator import generate_gait

class HumanoidWalking(Task):
    """Locomotion with the Unitree G1 humanoid."""

    def __init__(self, planning_horizon, sim_steps_per_control, target_vel, weights: Weights, gait_settings: GaitSettings):
        """Load the MuJoCo model and set task parameters."""
        # TODO: Go back to this xml once I have figured out why it is slow
        # mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/g1_small_minimal_contacts.xml")
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/basic_scene.xml")

        print("---- Humanoid walking task ----\nHorizon: " + str(planning_horizon) + "\nSim steps per control: "
              + str(sim_steps_per_control) + "\nTarget vel: " + str(target_vel) + "\n")

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control,
            trace_sites=["imu", "left_foot", "right_foot"],
        )

        # Get sensor and site ids
        self.orientation_sensor_id = mj_model.sensor("imu-body-quat").id
        self.velocity_sensor_id = mj_model.sensor("imu-body-linvel").id
        self.torso_id = mj_model.site("imu").id     # This really the waist

        # Set the target velocity (m/s) and height
        self.target_velocity = target_vel #0.0
        self.target_height = 0.9

        # Weights
        self.weights = weights
        self.weights.print_weights()

        # TODO: Change to toes and heels
        # Feet
        self.right_foot_sites = ["right_toe", "right_heel"]
        self.left_foot_sites = ["left_toe", "left_heel"]
        # Get site IDs for feet
        self.right_foot_site_ids = jnp.array(
            [mj_model.site(name).id for name in self.right_foot_sites]
        )
        self.left_foot_site_ids = jnp.array(
            [mj_model.site(name).id for name in self.left_foot_sites]
        )

        self.gait_settings = gait_settings

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        return state.site_xpos[self.torso_id, 2]

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        """Get the rotation from the current torso orientation to upright."""
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(quat, goal_quat)

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.velocity_sensor_id]
        return state.sensordata[sensor_adr]

    def _get_foot_height(self, foot_id: int, state: mjx.Data) -> jax.Array:
        # jax.debug.print("foot height: {x}", x=state.site_xpos[foot_id][2])
        return state.site_xpos[foot_id]

    def _get_des_foot_height(self, time_offset: float, time: float) -> float:
        # jax.debug.print("time: {x}", x=time)
        # time_into_gait =  (time - (self.gait_settings.start_time - time_offset)) % (
        #             self.gait_settings.swing_time*2)  # For now always assuming two distinct swings
        time_into_gait = (time) % (self.gait_settings.swing_time*2)
        # jax.debug.print("time into gait: {x}", x=time_into_gait)
        return generate_gait(time_into_gait, self.gait_settings)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )

        velocity_cost = jnp.square(
            self._get_torso_velocity(state) - self.target_velocity
        )

        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )

        control_cost = jnp.sum(jnp.square(control))

        foot_cost = 0
        for foot_id in self.right_foot_site_ids:
            foot_cost += jnp.square(self._get_foot_height(foot_id, state)[2] - self._get_des_foot_height(0, state.time))

        time_offset = self.gait_settings.swing_time
        for foot_id in self.left_foot_site_ids:
            foot_cost += jnp.square(self._get_foot_height(foot_id, state)[2] - self._get_des_foot_height(time_offset, state.time))
            # jax.debug.print("des foot height: {x}, current height: {y}", x=self._get_des_foot_height(time_offset, state.time), y=self._get_foot_height(foot_id, state)[2])

        # foot_cost += jnp.square(self._get_foot_height(self.feet_site_ids[0], state)[2] - self._get_des_foot_height(time_offset, state.time))

        return (
            self.weights.orientation_tracking * orientation_cost
            + self.weights.velocity_tracking * velocity_cost
            + self.weights.height_tracking * height_cost
            + self.weights.control_cost * control_cost
            + self.weights.gait_tracking * foot_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        return orientation_cost + 10 * height_cost
