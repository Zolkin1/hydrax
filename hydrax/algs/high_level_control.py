from abc import ABC, abstractmethod
from functools import partial
from typing import Literal, Tuple, Any
from flax.struct import dataclass

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, Trajectory, SamplingParams
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


class LowLevelController:
    """A low level controller"""

    def __init__(self, action_size: int):
        """Initialize the controller"""
        self.action_size = action_size

    @abstractmethod
    def compute_control(
        self, state: mjx.Data, u_high_level: jax.Array, data: jax.Array,
    ) -> jax.Array:
        """Compute the control action"""

@dataclass
class HLSamplingParams(SamplingParams):
    """Parameters for sampling-based control algorithms.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

    ll_data: jax.Array

class HighLevelControl(SamplingBasedController, ABC):
    """A class that controls the robot through a high level controller given a low level controller."""

    def __init__(
        self,
        task: Task,
        controller: LowLevelController,
        num_inputs: int,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        plan_horizon: float,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller"""
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

        self.controller = controller
        self.num_inputs = num_inputs

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, _: Any):
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, nu)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations_data(
                state, new_tk, knots, params.ll_data, dr_rng
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        return params, rollouts_final

    def rollout_with_randomizations_data(
            self,
            state: mjx.Data,
            tk: jax.Array,
            knots: jax.Array,
            data: jax.Array,
            rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None, None)
        )(self.model, states, controls, knots, data)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        data: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            data: Low level control data.

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(carry, u: jax.Array
        ) -> Tuple[Tuple[mjx.Data, jax.Array], Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x, data = carry
            action, data = self.controller.compute_control(x, u, data)
            x = x.replace(ctrl=action)
            x = mjx.step(model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return (x, data), (x, cost, sites)

        (final_state, final_data), (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, (state, data), controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def init_params(
            self, initial_knots: jax.Array = None, seed: int = 0
    ) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_knots: The initial knots of the control spline.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else jnp.zeros((self.num_knots, self.num_inputs))
        )
        assert mean.shape == (self.num_knots, self.num_inputs), (
            f"Initial knots must have shape (num_knots, nu), got {mean.shape}"
        )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)

        ll_data = jnp.zeros(self.controller.action_size)
        return HLSamplingParams(tk=tk, mean=mean, rng=rng, ll_data=ll_data)