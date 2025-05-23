import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
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

        
        self.decision_interval = 0.0525  
        self.steps_per_decision = int(self.decision_interval / self.timestep)
        self.step_counter = 0
        
        # --- Control signals ---
        self.control_signal = np.array([1.0, 1.0])
        self.prev_control_signal = np.array([1.0, 1.0])
        
        # --- Chemotaxis parameters ---
        self.attractive_gain = -500
        self.aversive_gain = 500
        self.delta_min = 0.2
        self.delta_max = 1.0

        # --- Vision system ---
        self.retina = Retina()
        self.coms = self.compute_ommatidia_centers()
        self.obj_threshold = 0.2  
        
        # --- Path integration ---
        self.returning = False
        self.velocity_hist = []
        self.heading_hist = []
        self.position = np.zeros(2)
        self.position_hist = []
        self.theta = 0.0
        self.count = 0
        self.distance_to_nest = 0.0
        self.angle_to_nest = 0.0
        self.theta_hist = []
        self.velocity_smoothing = 50
        
        # --- Escape "stuck" state ---
        self.STEP_BACK_STEPS = 1500
        self.STEP_BACK_COOLDOWN = 1500
        self.stuck_counter = 0
        self.stuck_cooldown = 0
    
    
    
    # --- METHODS ---
    
    # --- Visual information --- 
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

        features[:, 0] /= self.retina.nrows
        features[:, 1] /= self.retina.ncols
        features[:, 2] /= self.retina.num_ommatidia_per_eye
        return features.ravel()
    
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
    
    def compute_ball_signal(self, obs):
        raw_vision = obs.get("raw_vision", None)
        object_in_left = False
        object_in_right = False
        red_seen = False
        if raw_vision is not None:
            RED_THRESHOLD = 0.6
            GREEN_THRESHOLD = 0.3
            BLUE_THRESHOLD = 0.3
            RED_AREA_THRESHOLD = 0.005  

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
            delta_left = self.delta_max
            delta_right = self.delta_max

            if object_in_left and not object_in_right:
                delta_left *= 0.5
            elif object_in_right and not object_in_left:
                delta_right *= 0.5
            elif object_in_left and object_in_right:
                delta_left *= 0.8
                delta_right *= 0.2

            return red_seen, -np.array([delta_left, delta_right]) 
        else : 
            return red_seen, - np.array([0,0])
    
    # --- Odour --- 
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
    
    # --- Path integration --- 
    def set_returning(self, flag: bool):
        self.returning = flag
    
    def integrate_velocity(self, velocity, heading):       
        dt = self.timestep     
        if abs(velocity[0]) <= 15:
            velocity_x= 0
        else:
            velocity_x= velocity[0]
        world_x= velocity_x*np.cos(heading) * dt * self.velocity_smoothing
        world_y= velocity_x*np.sin(heading) * dt * self.velocity_smoothing
        if abs(world_x) < 0.05:
            world_x = 0
        if abs(world_y) < 0.05:
            world_y = 0
        self.position[0] += world_x
        self.position[1] += world_y
        self.theta = np.arctan2(self.position[1], self.position[0])
    
    def compute_nest(self, position, theta, smooth_heading):
        self.distance_to_nest = np.linalg.norm(position)
        self.angle_to_nest = np.clip(np.pi + theta - smooth_heading, -np.pi, np.pi)

    # --- Step back --- 
    def step_back(self):
        return np.array([-1.0, -1.0])
    
    def get_actions(self, obs: Observation) -> Action:
        
        raw_vision = obs.get("raw_vision", None)
        velocity = obs.get("velocity", None)
        heading = obs.get("heading", None)
        self.velocity_hist.append(velocity)
        self.heading_hist.append((heading + 2*np.pi) % (2 * np.pi))   
            
        if len(self.velocity_hist) % self.velocity_smoothing == 0:

            smooth_velocity = np.mean(self.velocity_hist[-10:], axis=0)
            smooth_heading = np.mean(np.unwrap(self.heading_hist)[-10:], axis=0)
            self.integrate_velocity(smooth_velocity,smooth_heading)
            self.position_hist.append(self.position.copy())
            self.theta_hist.append(self.theta)
            smooth_theta = np.unwrap(self.theta_hist)[-1]
            self.compute_nest(self.position, smooth_theta, smooth_heading)
            
        if obs.get("reached_odour", None) == False:
            if len(self.position_hist) >= 20:             
                if self.stuck_counter > 0:
                    #print("STEP BACK phase")
                    self.stuck_counter -= 1
                    control_signal = self.step_back()
                    joint_angles, adhesion = step_cpg(
                        cpg_network=self.cpg_network,
                        preprogrammed_steps=self.preprogrammed_steps,
                        action=control_signal,
                    )
                    return {"joints": joint_angles, "adhesion": adhesion}

                elif np.linalg.norm(self.position_hist[-19] - self.position_hist[-1]) < 0.5 and self.stuck_cooldown == 0:
                    #print("Stuck detected, initiating step_back")
                    self.stuck_counter = self.STEP_BACK_STEPS
                    self.stuck_cooldown = self.STEP_BACK_COOLDOWN
                    control_signal = self.step_back()
                    joint_angles, adhesion = step_cpg(
                        cpg_network=self.cpg_network,
                        preprogrammed_steps=self.preprogrammed_steps,
                        action=control_signal,
                    )
                    return {"joints": joint_angles, "adhesion": adhesion}

                elif self.stuck_cooldown > 0:
                    #print("Cooldown active, skipping step_back condition")
                    self.stuck_cooldown -= 1
                    #print("b'")
                    features = self.process_visual_obs(obs)
                    chemotaxis_signal = self.compute_chemotaxis_signal(obs)
                    #avoidance_steer = self.compute_avoidance_steer(features)
                    # Combine strategies
                    control_signal = 1.3 * chemotaxis_signal
                    red, ball_signal = self.compute_ball_signal(obs)
                    if red :
                        #print("red") 
                        control_signal = 0.2 * ball_signal
                    # Smooth
                    decay = 0.85
                    control_signal = decay * self.prev_control_signal + (1 - decay) * control_signal
                    if not red:
                        control_signal = np.clip(control_signal, 0.5, 1.5)
                    #control_signal = np.zeros(2)
                    self.prev_control_signal = control_signal
                    self.step_counter += 1
                    joint_angles, adhesion = step_cpg(
                        cpg_network=self.cpg_network,
                        preprogrammed_steps=self.preprogrammed_steps,
                        action=control_signal,
                    )
                    return {"joints": joint_angles, "adhesion": adhesion}  
                else:
                    #print("b")
                    features = self.process_visual_obs(obs)
                    chemotaxis_signal = self.compute_chemotaxis_signal(obs)
                    avoidance_steer = self.compute_avoidance_steer(features)
                    # Combine strategies
                    control_signal = 1.5 * chemotaxis_signal + avoidance_steer
                    red, ball_signal = self.compute_ball_signal(obs)
                    if red :
                        #print("red") 
                        control_signal = 0.2 * ball_signal
                    # Smooth
                    decay = 0.85
                    control_signal = decay * self.prev_control_signal + (1 - decay) * control_signal
                    if not red:
                        control_signal = np.clip(control_signal, 0.5, 1.5)
                    #control_signal = np.zeros(2)
                    self.prev_control_signal = control_signal
                    self.step_counter += 1
                    joint_angles, adhesion = step_cpg(
                        cpg_network=self.cpg_network,
                        preprogrammed_steps=self.preprogrammed_steps,
                        action=control_signal,
                    )
                    return {"joints": joint_angles, "adhesion": adhesion}                 
            else: 
                #print("c")
                features = self.process_visual_obs(obs)
                chemotaxis_signal = self.compute_chemotaxis_signal(obs)
                avoidance_steer = self.compute_avoidance_steer(features)
                # Combine strategies
                control_signal = 1.5 * chemotaxis_signal + avoidance_steer
                red, ball_signal = self.compute_ball_signal(obs)
                if red :
                    #print("red") 
                    control_signal = 0.2 * ball_signal
                # Smooth
                decay = 0.85
                control_signal = decay * self.prev_control_signal + (1 - decay) * control_signal
                if not red:
                    control_signal = np.clip(control_signal, 0.5, 1.5)
                #control_signal = np.zeros(2)
                self.prev_control_signal = control_signal
                self.step_counter += 1
                joint_angles, adhesion = step_cpg(
                    cpg_network=self.cpg_network,
                    preprogrammed_steps=self.preprogrammed_steps,
                    action=control_signal,
                )
                return {"joints": joint_angles, "adhesion": adhesion}
        else:
            if self.angle_to_nest > 0:
                delta_left = self.delta_max - self.angle_to_nest * (self.delta_max - self.delta_min)
                delta_right = self.delta_max
            else:
                delta_left = self.delta_max
                delta_right = self.delta_max - abs(self.angle_to_nest) * (self.delta_max - self.delta_min)           
            

            self.control_signal = np.array([delta_left, delta_right])
            if self.distance_to_nest < 0.5:
                print("Nest reached — stopping.")
                self.quit = True
            
            self.step_counter += 1
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=self.control_signal,
            )
            return {"joints": joint_angles, "adhesion": adhesion}
            
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