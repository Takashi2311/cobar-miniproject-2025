import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg


class Controller(BaseController):
    def __init__(self,timestep=1e-4,seed=0):
        
        from flygym.examples.locomotion import PreprogrammedSteps
        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        
        #ADDED BY DAVID
        self.timestep = timestep
        self.decision_interval = 0.05  # seconds
        self.steps_per_decision = int(self.decision_interval / self.timestep)
        self.step_counter = 0
        self.control_signal = np.array([1.0, 1.0])  # initial control signal
        self.attractive_gain = -500
        self.aversive_gain = 80
        self.delta_min = 0.2
        self.delta_max = 1.0
        
    def get_actions(self, obs: Observation) -> Action:
        if self.step_counter % self.steps_per_decision == 0:
            # Compute odor asymmetry
            odor_intensity = obs["odor_intensity"]
            # Assuming odor_intensity shape is (2, 4): 2 odors, 4 sensors (left/right antennae)
            # Reshape to (2, 2): 2 odors, left/right
            try:
                odor_intensity = odor_intensity.reshape(2, 4)  # Try (2, 4), since 2 odors × 4 sensors?
            except Exception as e:
                print("Failed reshape:", odor_intensity.shape)
                raise e
            
            attractive_intensity = odor_intensity[0]
            aversive_intensity = odor_intensity[1]

            # Compute ΔI for each odor
            delta_I_attractive = (attractive_intensity[0] - attractive_intensity[1]) / (
                (attractive_intensity[0] + attractive_intensity[1]) / 2 + 1e-6
            )
            delta_I_aversive = (aversive_intensity[0] - aversive_intensity[1]) / (
                (aversive_intensity[0] + aversive_intensity[1]) / 2 + 1e-6
            )

            # Compute s
            s = self.attractive_gain * delta_I_attractive + self.aversive_gain * delta_I_aversive

            # Compute turning bias b
            b = np.tanh(s ** 2) * np.sign(s)

            # Modulate control signals
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