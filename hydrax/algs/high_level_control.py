from abc import ABC, abstractmethod
from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


class LowLevelController:
    """A low level controller"""

    @abstractmethod
    def compute_control(
        self, u_high_level: jax.Array, state: mjx.Data
    ) -> jax.Array:
        """Compute the control action"""


class HighLevelControl(SamplingBasedController, ABC):
    """A class that controls the robot through a high level controller given a low level controller."""

    def __init__(
        self,
        task: Task,
        controller: LowLevelController,
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

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            u_ll = self.controller.compute_control(u, x)
            x = x.replace(ctrl=u_ll)
            x = mjx.step(model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
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
