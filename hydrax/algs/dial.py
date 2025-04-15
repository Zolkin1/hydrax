from typing import Literal, Tuple, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class DIALParams(SamplingParams):
    """Policy parameters for DIAL-MPC.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        # horizon_noise_sched: function defining the noise along the horizon. The function takes in the index along the
        #     horizon and the horizon length and outputs the noise.
        # iteration_noise_sched: function defining the noise along the iteration. The function takes in the iteration
        #     index and iteration length and outputs the noise.
    """

    # horizon_noise_sched: Callable[[int], jax.Array]
    # iteration_noise_sched: Callable[[int], jax.Array]

    beta_h: float
    beta_i: float

class DIAL(SamplingBasedController):
    """DIAL MPC.
    Implements the DIAL-MPC algorithm given in https://arxiv.org/abs/2409.15610
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        beta_h: float = 0.9,
        beta_i: float = 0.5,
    ) -> None:
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
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.num_samples = num_samples
        self.temperature = temperature
        self.beta_h = beta_h
        self.beta_i = beta_i

    def init_params(self, seed: int = 0) -> DIALParams:
        """Initialize the policy parameters."""
        _params = super().init_params(seed)

        return DIALParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, beta_h=self.beta_h, beta_i=self.beta_i)

    def sample_knots(self, params: DIALParams, iteration: int = 0) -> Tuple[jax.Array, DIALParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )

        # Compute the noise level
        horizon_noise = self.default_horizon_noise(params.beta_h, self.num_knots)          # Horizon is the length of the knots
        iteration_noise = self.default_iteration_noise(params.beta_i, self.iterations)     # Noise over the iterations

        # Combine the noise
        noise_level = iteration_noise[iteration] * horizon_noise
        noise_level = noise_level.reshape(-1, 1)

        controls = params.mean + noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: DIALParams, rollouts: Trajectory
    ) -> DIALParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)

    def default_horizon_noise(self, beta_h: float, H: int) -> jax.Array:
        du = len(self.task.u_max)
        horizon_idx = jnp.arange(0, H)
        return jnp.exp(-((H - horizon_idx)/(beta_h * H))*du)

    def default_iteration_noise(self, beta_i: float, N: int) -> jax.Array:
        du = len(self.task.u_max)
        iteration_idx = jnp.arange(0, N)
        return jnp.exp(-((N - iteration_idx)/(beta_i * N))*du)