import numpy as np
from cobar_miniproject.base_controller import Action, BaseController
from .utils import get_cpg, step_cpg

class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
        attractive_gain=-10,
        decision_interval=0.1  # in steps
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.step_counter = 0
        self.odor_memory = 0
        self.no_odor_counter = 0
                
        # Navigation parameters
        self.attractive_gain = attractive_gain
        self.decision_interval = decision_interval 
        self.last_decision_step = 0  # 统一使用step计数
        self.current_bias = 0  # [-1, 1] range for turning bias

    def get_actions(self, obs: dict) -> Action:  # 注意obs是dict
        # Make navigation decisions at fixed intervals
        if self.step_counter - self.last_decision_step >= self.decision_interval:
            self._update_navigation_decision(obs)
            self.last_decision_step = self.step_counter
        
        # Convert navigation bias to CPG modulation
        left_signal = 1.0 + min(0, self.current_bias) * 1.2
        right_signal = 1.0 - max(0, self.current_bias) * 1.2

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=np.array([left_signal, right_signal])
        )

        self.step_counter += 1
        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }
    
    def _update_navigation_decision(self, obs: dict):
        try:

            odor = obs["odor_intensity"]  # shape: (2, 4)

        # 按照方向索引（假设为 [front_left, front_right, back_left, back_right]）
            left_front = odor[0][0]
            left_back = odor[0][2]
            right_front = odor[1][1]
            right_back = odor[1][3]

        # X 方向梯度（左 vs 右）
            left_total = left_front + left_back
            right_total = right_front + right_back
            x_gradient = right_total - left_total

        # Y 方向梯度（前 vs 后）
            front_total = left_front + right_front
            back_total = left_back + right_back
            y_gradient = front_total - back_total

            angle = np.arctan2(y_gradient, x_gradient)
            attr_sum = left_total + right_total + 1e-6

            print("梯度：", x_gradient, y_gradient, "角度：", angle, "总强度：", attr_sum)

        # 控制逻辑（和你之前基本一致）...
            if abs(angle) < np.pi / 4:
                self.current_bias = 0.5
            elif abs(angle - np.pi) < np.pi / 4:
                self.current_bias = -0.5
            elif angle > np.pi / 2 or angle < -np.pi / 2:
                self.current_bias = np.random.choice([-0.8, 0.8])
            else:
                self.current_bias = 0.0

            if attr_sum > 0.05:
                self.current_bias *= 1.5

            if abs(x_gradient) > 1e-2 or abs(y_gradient) > 1e-2:
                self.odor_memory = np.arctan2(y_gradient, x_gradient)

            if attr_sum < 0.05:
                self.no_odor_counter += 1
            else:
                self.no_odor_counter = 0

        except Exception as e:
            print("气味解析错误：", e)
            self.current_bias = 0.0

    # 探索策略
        if self.no_odor_counter > 50:
            if self.odor_memory != 0:
                self.current_bias = 0.4 * np.sign(np.sin(self.odor_memory))
        else:
            self.current_bias = 0.5 * np.sin(self.step_counter / 100)
    

       

    def done_level(self, obs: dict) -> bool:
        try:
            return bool(obs.get("reached_odour", False))
        except Exception as e:
            print("ERROR checking done condition:", e)
            return False

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.step_counter = 0
        self.last_decision_step = 0
        self.current_bias = 0
        self.odor_memory = 0           
        self.no_odor_counter = 0       
  
        self.quit = False

