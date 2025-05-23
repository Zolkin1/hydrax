import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.low_level_controllers.rl_controller import RLController


class G1Controller(RLController):
    """G1 RL Policy"""

    def __init__(
        self,
        checkpoint_path,
        hidden_dims,
        obs_size,
        action_size,
        ang_vel_scale,
        default_angles,
        cmd_scale,
        qvel_scale,
        action_scale,
        period,
    ):
        """Initialize the RL controller"""
        super().__init__(
            obs_size=obs_size,
            action_size=action_size,
            hidden_dims=hidden_dims,
            checkpoint_path=checkpoint_path,
        )

        self.ang_vel_scale = ang_vel_scale
        self.default_angles = default_angles
        self.period = period
        self.cmd_scale = cmd_scale
        self.qvel_scale = qvel_scale
        self.action_scale = action_scale

        self.isaac_to_mujoco = {
            0: 0,  # left_hip_pitch
            1: 6,  # right_hip_pitch
            2: 12,  # waist_yaw
            3: 1,  # left_hip_roll
            4: 7,  # right_hip_roll
            5: 13,  # left_shoulder_pitch
            6: 17,  # right_shoulder_pitch
            7: 2,  # left_hip_yaw
            8: 8,  # right_hip_yaw
            9: 14,  # left_shoulder_roll
            10: 18,  # right_shoulder_roll
            11: 3,  # left_knee
            12: 9,  # right_knee
            13: 15,  # left_shoulder_yaw
            14: 19,  # right_shoulder_yaw
            15: 4,  # left_ankle_pitch
            16: 10,  # right_ankle_pitch
            17: 16,  # left_elbow
            18: 20,  # right_elbow
            19: 5,  # left_ankle_roll
            20: 11,  # right_ankle_roll
        }

        self.action_isaac = jnp.zeros(self.action_size)

    def create_obs(self, data: mjx.Data, input: jax.Array) -> jax.Array:
        """Convert mujoco data to the observation"""
        body_ang_vel = data.qvel[3:6]
        des_vel = input

        qj = data.qpos[7:] - self.default_angles

        obs = jnp.concatenate(
            [
                body_ang_vel * self.ang_vel_scale,
                self.compute_pg(data),
                jnp.array([des_vel[0] * self.cmd_scale[0]]),
                jnp.array([des_vel[1] * self.cmd_scale[1]]),
                jnp.array([des_vel[2] * self.cmd_scale[2]]),
                self.convert_to_isaac(qj),
                self.convert_to_isaac(data.qvel[6:]) * self.qvel_scale,
                self.action_isaac,
                jnp.array([jnp.sin(2 * jnp.pi * data.time / self.period)]),
                jnp.array([jnp.cos(2 * jnp.pi * data.time / self.period)]),
            ]
        )

        # obs = obs.at[:3].set(body_ang_vel * self.ang_vel_scale)  # Angular velocity
        # obs = obs.at[3:6].set(self.compute_pg(data))  # Projected gravity
        # obs = obs.at[6].set(des_vel[0] * self.cmd_scale[0])  # Command velocity
        # obs = obs.at[7].set(des_vel[1] * self.cmd_scale[1])  # Command velocity
        # obs = obs.at[8].set(des_vel[2] * self.cmd_scale[2])  # Command velocity
        #
        # nj = len(data.qpos) - 7
        #
        # qj = data.qpos[7:] - self.default_angles
        # obs = obs.at[9: 9 + nj].set(self.convert_to_isaac(qj))  # Joint pos
        # obs = obs.at[9 + nj: 9 + 2 * nj].set(self.convert_to_isaac(data.qvel[6:]) * self.qvel_scale)  # Joint vel
        #
        #
        # obs = obs.at[9 + 2 * nj: 9 + 3 * nj].set(self.action_isaac)  # Past action
        #
        # sin_phase = jnp.sin(2 * jnp.pi * data.time / self.period)
        # cos_phase = jnp.cos(2 * jnp.pi * data.time / self.period)
        # obs = obs.at[9 + 3 * nj: 9 + 3 * nj + 2].set(jnp.array([sin_phase, cos_phase]))  # Phases

        return obs

    def create_action(self, obs: jax.Array) -> jax.Array:
        """Get action from RL Policy"""
        self.action_isaac = self.model(obs)

        return (
            self.convert_to_mujoco(self.action_isaac) * self.action_scale
            + self.default_angles
        )

    def compute_pg(self, data: mjx.Data) -> jax.Array:
        """Compute the projected gravity"""
        qw, qx, qy, qz = data.qpos[3:7]

        return jnp.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ]
        )

    def convert_to_isaac(self, vec):
        isaac_vec = jnp.zeros(self.action_size)
        isaac_indices = jnp.array(list(self.isaac_to_mujoco.keys()))
        mujoco_indices = jnp.array(list(self.isaac_to_mujoco.values()))
        return isaac_vec.at[isaac_indices].set(vec[mujoco_indices])

    def convert_to_mujoco(self, vec):
        mj_vec = jnp.zeros(self.action_size)
        isaac_indices = jnp.array(list(self.isaac_to_mujoco.keys()))
        mujoco_indices = jnp.array(list(self.isaac_to_mujoco.values()))
        return mj_vec.at[mujoco_indices].set(vec[isaac_indices])
