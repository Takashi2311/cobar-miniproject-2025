import numpy as np
from cobar_miniproject.base_controller import Action, BaseController
from .utils import get_cpg, step_cpg

class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0, decision_interval=0.05, smoothing_factor=0.2):
        from flygym.examples.locomotion import PreprogrammedSteps
        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.step_counter = 0
        self.odor_memory = 0
        self.no_odor_counter = 0

        # 控制参数
        self.attractive_gain = -500
        self.decision_interval = decision_interval
        self.last_decision_step = 0
        self.current_bias = 0
        self.last_bias = 0
        self.smoothing_factor = smoothing_factor

    def get_actions(self, obs: dict) -> Action:
        # 决策时机
        if self.step_counter - self.last_decision_step >= self.decision_interval:
            self._update_navigation_decision(obs)
            self.last_decision_step = self.step_counter

        # 平滑的CPG转向信号
        bias = np.clip(self.current_bias, -1.0, 1.0)
        left_signal = 1.0 + 0.8 * np.tanh(min(0, bias))   # 向左转时增强左侧
        right_signal = 1.0 - 0.8 * np.tanh(max(0, bias))  # 向右转时增强右侧

        # 脚步生成
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=np.array([left_signal, right_signal])
        )

        self.step_counter += 1
        return {"joints": joint_angles, "adhesion": adhesion}

    def _update_navigation_decision(self, obs: dict):
        
        try:
            odor = obs["odor_intensity"][0]  # 使用第一个气味通道（吸引性）

            # reshape 为 (2, 2): 左右 x 前后
            reshaped = odor.reshape(2, 2)
            weighted_lr = np.average(reshaped, axis=0, weights=[9, 1])  # 左右方向重加权
            odor_diff = weighted_lr[0] - weighted_lr[1]
            mean_odor = weighted_lr.mean() + 1e-6

            # 应用导航偏置
            effective_bias = self.attractive_gain * (odor_diff / mean_odor)

            # tanh平滑 + square增强非线性
            target_bias = np.tanh(effective_bias ** 2) * np.sign(effective_bias)

            # 应用平滑
            self.current_bias = (
                self.smoothing_factor * target_bias + (1 - self.smoothing_factor) * self.last_bias
            )
            self.last_bias = self.current_bias

        except Exception as e:
            print("⚠️ 气味导航决策错误：", e)
            self.current_bias = 0.0

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
        self.last_bias = 0
        self.odor_memory = 0
        self.no_odor_counter = 0
        self.quit = False
