from dataclasses import dataclass

import yaml


@dataclass
class Weights:
    orientation_tracking: float
    velocity_tracking: float
    height_tracking: float
    control_cost: float
    gait_tracking: float

    def print_weights(self) -> None:
        print("---- Weights ----\nOrientation tracking: " + str(self.orientation_tracking)
              + "\nVelocity tracking: " + str(self.velocity_tracking) + "\nHeight tracking: " + str(self.height_tracking)
              + "\nControl cost: " + str(self.control_cost) + "\nGait tracking: " + str(self.gait_tracking) + "\n")

@dataclass
class GaitSettings:
    apex_height: float
    apex_time: float
    ground_height: float
    swing_time: float
    start_time: float

@dataclass
class Settings:
    ctrl_freq: float
    planning_horizon: int
    sim_steps_per_control: int
    num_samples: int
    target_vel: float
    weights: Weights
    gait: GaitSettings
    noise_level: float = 0.2
    temperature: float = 0.1
    sigma_start: float = 0.2
    sigma_min: float = 0.2
    num_elite: int = 5

def read_settings(file_path: str) -> Settings:
    """
    Read in settings for both the controller and the task from a yaml file.
    """
    with open(file_path, 'r') as f:
        settings_file = yaml.safe_load(f)

    weights = Weights(settings_file['weights']['orientation_tracking'],
                    settings_file['weights']['velocity_tracking'],
                    settings_file['weights']['height_tracking'],
                    settings_file['weights']['control_cost'],
                    settings_file['weights']['gait_tracking'],)

    gait = GaitSettings(settings_file['gait']['apex_height'],
                        settings_file['gait']['apex_time'],
                        settings_file['gait']['ground_height'],
                        settings_file['gait']['swing_time'],
                        settings_file['gait']['start_time'],)

    settings = Settings(settings_file['ctrl_params']['ctrl_freq'],
                        settings_file['ctrl_params']['planning_horizon'],
                        settings_file['ctrl_params']['sim_steps_per_control'],
                        settings_file['sample_params']['num_samples'],
                        settings_file['ctrl_params']['target_vel'],
                        weights,
                        gait)

    if 'noise_level' in settings_file['ctrl_params']:
        settings.noise_level = settings_file['ctrl_params']['noise_level']

    if 'temperature' in settings_file['ctrl_params']:
        settings.temperature = settings_file['ctrl_params']['temperature']

    if 'sigma_start' in settings_file['ctrl_params']:
        settings.sigma_start = settings_file['ctrl_params']['sigma_start']

    if 'sigma_min' in settings_file['ctrl_params']:
        settings.sigma_min = settings_file['ctrl_params']['sigma_min']

    if 'num_elite' in settings_file['ctrl_params']:
        settings.num_elite = settings_file['ctrl_params']['num_elite']

    return settings
