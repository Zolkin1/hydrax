import jax.numpy as jnp

import matplotlib.pyplot as plt

from gait_generator import generate_gait
from hydrax.read_settings import GaitSettings


def test_gait(gait_settings: GaitSettings) -> None:
    times = jnp.linspace(0, gait_settings.swing_time*2, 100)
    heights = []
    for t in times:
        heights.append(generate_gait(t, gait_settings))

    _, ax = plt.subplots()

    ax.plot(times, heights)
    ax.set_ylabel(r"$z$")
    ax.set_xlabel(r"$t$")
    plt.show()

if __name__ == "__main__":
    gait_settings = GaitSettings(0.1, 0.75, 0.02, 0.3, 0)
    test_gait(gait_settings)