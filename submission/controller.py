import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from gymnasium import spaces
from .utils import get_cpg, step_cpg
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.vision import Retina

class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        super().__init__()
        self.quit = False
        self.timestep = timestep
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()

        # ADDED BY DAVID
        self.decision_interval = 0.00525  # seconds
        self.steps_per_decision = int(self.decision_interval / self.timestep)
        self.step_counter = 0
        
        # Control signals
        self.control_signal = np.array([1.0, 1.0])
        self.prev_control_signal = np.array([1.0, 1.0])
        
        # Chemotaxis parameters
        self.attractive_gain = -500
        self.aversive_gain = 500
        self.delta_min = 0.2
        self.delta_max = 1.0

        # ADDED BY TAKASHI
        self.obj_threshold = 0.2  # brightness threshold
        self.area_threshold = 0.01  # fraction of dark ommatidia to trigger avoidance
        
        # Vision system
        self.retina = Retina()
        self.coms = self.compute_ommatidia_centers()
        
    def compute_ommatidia_centers(self):
        coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            coms[i, :] = np.argwhere(mask).mean(axis=0)
        return coms

    def process_visual_obs(self, raw_obs):
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(raw_obs["vision"]):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]

        # Normalize
        features[:, 0] /= self.retina.nrows
        features[:, 1] /= self.retina.ncols
        features[:, 2] /= self.retina.num_ommatidia_per_eye
        return features.ravel()

    def compute_chemotaxis_signal(self, obs):
        odor_intensity = obs["odor_intensity"].reshape(2, 4)
        attractive, aversive = odor_intensity

        delta_I_attr = (attractive[0] - attractive[1]) / ((attractive[0] + attractive[1]) / 2 + 1e-6)
        delta_I_avers = (aversive[0] - aversive[1]) / ((aversive[0] + aversive[1]) / 2 + 1e-6)

        s = self.attractive_gain * delta_I_attr + self.aversive_gain * delta_I_avers
        b = np.tanh(s ** 2) * np.sign(s)

        if b > 0:
            left = self.delta_max
            right = self.delta_max - b * (self.delta_max - self.delta_min)
        else:
            left = self.delta_max - abs(b) * (self.delta_max - self.delta_min)
            right = self.delta_max

        return np.array([left, right])

    def compute_avoidance_steer(self, features):
        left_eye = {"x": features[1], "area": features[2]}
        right_eye = {"x": features[4], "area": features[5]}

        steer_gain = 4
        min_area = 0.01
        dead_zone_x = 0.05

        steer = np.zeros(2)
        for eye in [left_eye, right_eye]:
            if eye["area"] > min_area:
                offset = eye["x"] - 0.5
                if abs(offset) > dead_zone_x:
                    correction = steer_gain * offset
                    steer[0] += correction
                    steer[1] -= correction
        return steer
    
    def get_actions(self, obs: Observation) -> Action:
        raw_vision = obs.get("raw_vision", None)
        object_in_left = False
        object_in_right = False
        red_seen = False
        if raw_vision is not None:
            RED_THRESHOLD = 0.6
            GREEN_THRESHOLD = 0.3
            BLUE_THRESHOLD = 0.3
            RED_AREA_THRESHOLD = 0.005  # 0.5%
            for eye in [0, 1]:
                img = raw_vision[eye]  # (512, 450, 3)
                red_mask = (
                    (img[:, :, 0] > RED_THRESHOLD) &
                    (img[:, :, 1] < GREEN_THRESHOLD) &
                    (img[:, :, 2] < BLUE_THRESHOLD)
                )
                area_ratio = np.sum(red_mask) / (img.shape[0] * img.shape[1])
                if eye == 0:
                    object_in_left = area_ratio > RED_AREA_THRESHOLD
                else:
                    object_in_right = area_ratio > RED_AREA_THRESHOLD
            red_seen = object_in_left or object_in_right
        if red_seen:
            print("Red object detected — retreating.")
            delta_left = self.delta_max
            delta_right = self.delta_max
            if object_in_left and not object_in_right:
                delta_left *= 0.5
            elif object_in_right and not object_in_left:
                delta_right *= 0.5
            elif object_in_left and object_in_right:
                delta_left *= 0.8
                delta_right *= 0.2
            self.control_signal = -np.array([delta_left, delta_right])
            self.step_counter += 1
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=self.control_signal,
            )
            return {"joints": joint_angles, "adhesion": adhesion}
        features = self.process_visual_obs(obs)
        chemotaxis_signal = self.compute_chemotaxis_signal(obs)
        avoidance_steer = self.compute_avoidance_steer(features)

        # Combine strategies
        control_signal = 1.5 * chemotaxis_signal + avoidance_steer

        # Smooth
        decay = 0.9
        control_signal = decay * self.prev_control_signal + (1 - decay) * control_signal
        control_signal = np.clip(control_signal, 0.5, 1.5)

        self.prev_control_signal = control_signal
        self.step_counter += 1

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=control_signal,
        )

        return {"joints": joint_angles, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return obs.get("reached_odour", False)

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.step_counter = 0
        self.control_signal = np.array([1.0, 1.0])
