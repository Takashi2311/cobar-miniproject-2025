from abc import ABC, abstractmethod
import numpy as np
from typing import TypedDict


class Observation(TypedDict, total=False):
    """
    The observations from the environment. Namely:
    - `"joints"` - the angle of each of the fly's joints
    - `"end_effectors"` - the end effector positions in egocentric coordinates
    - `"contact_forces"` - the contact forces
    - `"heading"` - the fly's absolute heading
    - `"velocity"` - the fly's velocity along x, y, z
    - `"odor_intensity"` - the odour intensity in each antenna
    - `"vision"` - fly-like vision image
    - `"raw_vision"` - raw vision image
    - `"vision_updated"` - whether the fly's vision was updated in the last simulation step
    - `"reached_odour"` - whether the fly has reached the odour source yet (used for the final level)
    """
    joints: np.ndarray
    end_effectors: np.ndarray
    contact_forces: np.ndarray
    heading: float
    velocity: np.ndarray
    odor_intensity: np.ndarray
    vision: np.ndarray
    raw_vision: np.ndarray
    vision_updated: bool
    reached_odour: bool


class Action(TypedDict):
    """
    The fly's joint angles under the "joints" key and
    the adhesion state for each leg under the "adhesion" key.
    """
    joints: np.ndarray
    adhesion: np.ndarray


class BaseController(ABC):
    def __init__(self, **kwargs):
        """
        Initialize the CobarController with any necessary parameters.
        """

    @abstractmethod
    def get_actions(self, obs: Observation) -> Action:
        """
        The heart of your controller: goes from observations to joint angles.

        Parameters
        ----------
        obs: Observation
            The observations from the environment. Namely:
            - `"joints"` - the angle of each of the fly's joints
            - `"end_effectors"` - the end effector positions in egocentric coordinates
            - `"contact_forces"` - the contact forces
            - `"heading"` - the fly's absolute heading
            - `"velocity"` - the fly's velocity along x, y, z
            - `"odor_intensity"` - the odour intensity in each antenna
            - `"vision"` - fly-like vision image
            - `"raw_vision"` - raw vision image
            - `"vision_updated"` - whether the fly's vision was updated in the last simulation step
            - `"reached_odour"` - whether the fly has reached the odour source yet (used for the final level)


        Returns
        -------
        action: Action
            The fly's joint angles under the "joints" key
            The adhesion state under the "adhesion" key
        """

        action: Action = {
            "joints": np.zeros(42),
            "adhesion": np.zeros(6),
        }
        return action

    @abstractmethod
    # def done_level(self, obs) -> bool:
    #     """
    #     For level 4 (path integration) - if the fly has returned back home
    #     after collecting the odour, return `True` here to stop the simulation.
    #     """
    #     pass
    def done_level(self, obs):
        # check if quit is set to true
        if self.quit:
            return True
        # check if the simulation is done
        return False

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the controller to its initial state.
        """
        pass
