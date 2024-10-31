import jax
import jax.numpy as jnp

from hydrax.read_settings import GaitSettings

def generate_gait(time, gait_settings: GaitSettings) -> float:
    """
    Computes the desired height of the foot given the current time and the gait settings.
    """
    # Verify apex time is between 0 and 1
    if gait_settings.apex_time <= 0 or gait_settings.apex_time >= 1:
        raise ValueError("Apex time must be 0 < apex_time < 1")

    # Determine if we are before or after the apex
    apex_abs = gait_settings.apex_time*gait_settings.swing_time

    condlist = [jnp.asarray(time < apex_abs, dtype=bool),
                jnp.asarray((time <= gait_settings.swing_time) & (time > apex_abs), dtype=bool),
                jnp.asarray(time > gait_settings.swing_time, dtype=bool)]
    funclist = [(gait_settings.ground_height
                - apex_abs**-2 * 3 * (gait_settings.ground_height - gait_settings.apex_height) * time**2
                + apex_abs**-3 * 2 * (gait_settings.ground_height - gait_settings.apex_height) * time**3),
                (gait_settings.apex_height
                 - (gait_settings.swing_time - apex_abs)**-2 * 3 * (
                             gait_settings.apex_height - gait_settings.ground_height) * (time - apex_abs)**2
                 + (gait_settings.swing_time - apex_abs)**-3 * 2 * (
                             gait_settings.apex_height - gait_settings.ground_height) * (time - apex_abs)**3),
                gait_settings.ground_height
                ]
    height = jnp.piecewise(time, condlist, funclist)
    # jax.debug.print("time: {x}, height: {y}", x=time, y=height)
    return height