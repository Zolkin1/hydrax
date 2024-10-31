from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class DialMPCParams:
    """Policy parameters for model-predictive path integral control.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    rng: jax.Array


class DialMPC(SamplingBasedController):
    """Dial MPC.

    Implements https://arxiv.org/abs/2409.15610.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        inner_loop_iterations: int,
        planning_horizon: int,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.inner_loop_iterations = inner_loop_iterations
        self.planning_horizon = planning_horizon

    def init_params(self, seed: int = 0) -> DialMPCParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon - 1, self.task.model.nu))
        return DialMPCParams(mean=mean, rng=rng)

    def sample_controls(
        self, params: DialMPCParams, trajectory: Trajectory, inner_iter: int,
    ) -> Tuple[jax.Array, DialMPCParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.task.planning_horizon - 1,
                self.task.model.nu,
            ),
        )
        # Get the covariance, vectorized over the horizon and samples
        cov = self.get_covariance(inner_iter)
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def get_covariance(self, shape: Tuple, inner_iter: int) -> jax.Array:
        """
        Get the covariance defined by the annealing strategy.
        Returns an array of the specified shape. The array is assumed to be of shape: sample x horizon x inputs.
        The covariance is constant across samples and inputs.
        """
        # Return a 1D array of length horizon
        cov = jnp.array(self.planning_horizon)
        for c in cov:
            c = jnp.exp(-((self.inner_loop_iterations - inner_iter)/(self.beta1 * self.inner_loop_iterations))
                        -(self.planning_horizon - i)/(self.beta2*self.planning_horizon))

    def update_params(
        self, params: DialMPCParams, rollouts: Trajectory
    ) -> DialMPCParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.controls, axis=0)
        return params.replace(mean=mean)

    def get_action(self, params: DialMPCParams, t: float) -> jax.Array:
        """Get the control action for the current time step."""
        # TODO: Add in more interpolation options
        # Linear
        # times = jnp.arange(0, self.task.dt*(self.task.planning_horizon-1), self.task.dt)
        # mean_interp = jax.vmap(lambda y_dim: jnp.interp(t, times, y_dim))(params.mean.T)
        #
        # return mean_interp.T
        # ZOH
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.mean[idx]
