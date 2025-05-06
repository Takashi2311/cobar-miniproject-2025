import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from gymnasium import spaces
from .utils import get_cpg, step_cpg


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps
        super().__init__()
        self.quit = False
        self.timestep = timestep
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()

        # ADDED BY DAVID
        self.decision_interval = 0.05  # seconds
        self.steps_per_decision = int(self.decision_interval / self.timestep)
        self.step_counter = 0
        self.control_signal = np.array([1.0, 1.0])
        self.attractive_gain = -500
        self.aversive_gain = 80
        self.delta_min = 0.2
        self.delta_max = 1.0

        # ADDED BY TAKASHI
        self.obj_threshold = 0.15  # brightness threshold
        self.area_threshold = 0.01  # fraction of dark ommatidia to trigger avoidance

    def get_actions(self, obs: Observation) -> Action:
        if self.step_counter % self.steps_per_decision == 0:
            vision = obs["vision"]  # shape: (2, 721, 2)
            brightness = vision.max(axis=-1)  # shape: (2, 721)
            dark_mask = brightness < self.obj_threshold
            area = dark_mask.sum(axis=1) / 721  # shape: (2,)
            raw_vision = obs["raw_vision"] # shape: (2, 512, 450, 3)
            print("raw_vision shape:", raw_vision.shape)

            object_in_left = area[0] > self.area_threshold
            object_in_right = area[1] > self.area_threshold
            object_seen = object_in_left or object_in_right
            if object_seen:
                print("Object detected — switching to escape mode.")
                delta_left = self.delta_max
                delta_right = self.delta_max

                if object_in_left and not object_in_right:
                    delta_left *= 0.5
                elif object_in_right and not object_in_left:
                    delta_right *= 0.5
                elif object_in_left and object_in_right:
                    delta_left *= 0.8
                    delta_right *= 0.8 

                self.control_signal = -np.array([delta_left, delta_right])

            else:
                odor_intensity = obs["odor_intensity"]
                try:
                    odor_intensity = odor_intensity.reshape(2, 4)
                except Exception as e:
                    print("Failed to reshape odor_intensity:", e)
                    raise e

                attractive = odor_intensity[0]
                aversive = odor_intensity[1]

                delta_I_attr = (attractive[0] - attractive[1]) / ((attractive[0] + attractive[1]) / 2 + 1e-6)
                delta_I_avers = (aversive[0] - aversive[1]) / ((aversive[0] + aversive[1]) / 2 + 1e-6)

                s = self.attractive_gain * delta_I_attr + self.aversive_gain * delta_I_avers
                b = np.tanh(s ** 2) * np.sign(s)

                if b > 0:
                    delta_left = self.delta_max
                    delta_right = self.delta_max - b * (self.delta_max - self.delta_min)
                else:
                    delta_left = self.delta_max - abs(b) * (self.delta_max - self.delta_min)
                    delta_right = self.delta_max

                self.control_signal = np.array([delta_left, delta_right])

        self.step_counter += 1

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=self.control_signal,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }


    def done_level(self, obs: Observation):
        return obs.get("reached_odour", False)

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.step_counter = 0
        self.control_signal = np.array([1.0, 1.0])
