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

            if raw_vision is not None:
                def extract_black_ratio(img):
                    H, W, _ = img.shape
                    center_col = W // 2
                    region = img[:, center_col - 15:center_col + 15, :]
                    R = region[:, :, 0]
                    G = region[:, :, 1]
                    B = region[:, :, 2]
                    black_mask = (R < 0.2) & (G < 0.2) & (B < 0.2)
                    return black_mask.sum() / black_mask.size
                
                def extract_left_black_ratio(img):
                    H, W, _ = img.shape
                    center_height = H // 4
                    target_width = int(17/145 * W)
                    region = img[:, -target_width:, :]
                    R = region[:, :, 0]
                    G = region[:, :, 1]
                    B = region[:, :, 2]
                    black_mask = (R < 0.2) & (G < 0.2) & (B < 0.2)
                    return black_mask.sum() / black_mask.size
                
                def extract_right_black_ratio(img):
                    H, W, _ = img.shape
                    center_height = H // 4
                    target_width = int(17/145 * W)
                    region = img[:, :target_width, :]
                    R = region[:, :, 0]
                    G = region[:, :, 1]
                    B = region[:, :, 2]
                    black_mask = (R < 0.2) & (G < 0.2) & (B < 0.2)
                    return black_mask.sum() / black_mask.size
                
                left_black_ratio = extract_left_black_ratio(raw_vision[0])
                right_black_ratio = extract_right_black_ratio(raw_vision[1])

                black_diff = right_black_ratio - left_black_ratio
                black_ratio = (right_black_ratio + left_black_ratio) /2
                BLACK_RATIO_THRESHOLD = 0.3
                SINGLE_BLACK_RATIO_THRESHOLD = 0.35
                print("Control signal:", self.control_signal)
                print("Left black ratio:", left_black_ratio)
                print("Right black ratio:", right_black_ratio)
                print("Black diff:", black_diff)
                print("Black ratio:", black_ratio)
                THRESH = 0.02            
                # if abs(black_diff) > THRESH:
                #     print("Black pillar detected — steering to avoid")
                #     if black_diff > 0:
                #         delta_left = self.delta_max
                #         delta_right = self.delta_min
                #     else:
                #         delta_left = self.delta_min
                #         delta_right = self.delta_max
                if abs(black_ratio) > BLACK_RATIO_THRESHOLD:
                    print("Black pillar detected — steering to avoid")
                    if black_diff > 0.002:# and right_black_ratio > SINGLE_BLACK_RATIO_THRESHOLD:
                        delta_left = self.delta_min*left_black_ratio
                        delta_right = self.delta_max*right_black_ratio
                    elif black_diff < -0.002:# and left_black_ratio > SINGLE_BLACK_RATIO_THRESHOLD:
                        delta_left = self.delta_max*left_black_ratio
                        delta_right = self.delta_min*right_black_ratio
                    else:
                        delta_left = -np.sign(black_diff)*self.delta_min
                        delta_right = np.sign(black_diff)*self.delta_min

                    self.control_signal = np.array([delta_left, delta_right])


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

        return {"joints": joint_angles, "adhesion": adhesion}

    # def done_level(self, obs: Observation):
    #     return obs.get("reached_odour", False)
    def done_level(self, obs):
        # check if quit is set to true
        if self.quit:
            return True
        # check if the simulation is done
        return False

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.step_counter = 0
        self.control_signal = np.array([1.0, 1.0])