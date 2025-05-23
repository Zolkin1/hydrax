from abc import abstractmethod
from typing import Sequence

import jax.numpy as jnp
import jax.random
import torch
from flax import nnx
from mujoco import mjx

from hydrax.algs.high_level_control import LowLevelController


def load_linear_weights(nnx_layer, weight, bias):
    # PyTorch uses (out, in) for Linear, Flax uses (in, out)
    nnx_layer.kernel.value = jnp.array(weight.T)
    nnx_layer.bias.value = jnp.array(bias)


def load_pytorch_weights(nnx_mlp, np_weights, prefix):
    layer_idx = 0
    for layer in nnx_mlp.layers:
        if isinstance(layer, nnx.Linear):
            w_key = f"{prefix}.{layer_idx}.weight"
            b_key = f"{prefix}.{layer_idx}.bias"
            if w_key in np_weights and b_key in np_weights:
                load_linear_weights(layer, np_weights[w_key], np_weights[b_key])
            layer_idx += 2  # Skip activation layers in nnx


class MLP(nnx.Module):
    def __init__(
        self,
        obs_size,
        action_size,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
        activation=nnx.elu,
    ):
        super().__init__()
        self.layers = []

        self.num_hidden = len(hidden_dims)
        self.activation = activation

        self.layers.append(nnx.Linear(obs_size, hidden_dims[0], rngs=rngs))

        for i in range(1, len(hidden_dims)):
            self.layers.append(
                nnx.Linear(hidden_dims[i - 1], hidden_dims[i], rngs=rngs)
            )

        self.layers.append(nnx.Linear(hidden_dims[-1], action_size, rngs=rngs))

    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class Policy(nnx.Module):
    def __init__(
        self,
        obs_size,
        action_size,
        actor_hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
        activation=nnx.elu,
    ):
        super().__init__()
        self.actor = MLP(
            obs_size, action_size, actor_hidden_dims, rngs, activation
        )

    def __call__(self, x):
        policy_out = self.actor(x)
        return policy_out


class RLController(LowLevelController):
    """RL Controller Wrapper"""

    def __init__(
        self,
        checkpoint_path,
        hidden_dims,
        obs_size,
        action_size,
    ):
        """Initialize the RL controller"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.obs_size = obs_size
        self.action_size = action_size

        self.model = Policy(
            obs_size=obs_size,
            action_size=action_size,
            actor_hidden_dims=hidden_dims,
            rngs=nnx.Rngs(0),
        )

        # Transfer the weights
        state_dict = checkpoint["model_state_dict"]
        np_weights = {k: v.numpy() for k, v in state_dict.items()}
        load_pytorch_weights(self.model.actor, np_weights, prefix="actor")

    @abstractmethod
    def create_obs(self, data: mjx.Data, input: jax.Array) -> jax.Array:
        """Create the observation from the mujoco data"""

    @abstractmethod
    def create_action(self, obs: jax.Array) -> jax.Array:
        """Create the action from the output of the network"""
