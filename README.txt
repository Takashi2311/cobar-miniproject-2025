# Controller Module Documentation

This module defines the `Controller` class used for simulated fly navigation in the Cobar MiniProject. The controller integrates multiple sensory modalities (olfaction, vision, path integration) to perform goal-directed locomotion with obstacle avoidance and homing.

## Class: `Controller`

### `__init__(self, timestep=1e-4, seed=0)`
Initializes the controller with parameters for locomotion, sensory processing, and path integration.

- **Args:**
  - `timestep` (`float`): Simulation timestep.
  - `seed` (`int`): Random seed for reproducibility.

---

### `compute_ommatidia_centers(self)`
Computes the spatial center of each ommatidium (visual unit) in the retina.

- **Returns:**
  - `coms` (`np.ndarray`): Coordinates of ommatidia centers.

---

### `process_visual_obs(self, raw_obs)`
Processes visual inputs to extract normalized features (mean x/y and area) of detected objects.

- **Observation keys used:** `"vision"`
- **Args:**
  - `raw_obs` (`dict`): Raw visual input.
- **Returns:**
  - `features` (`np.ndarray`): Array of shape (6,) — (left_x, left_y, left_area, right_x, right_y, right_area).

---

### `compute_chemotaxis_signal(self, obs)`
Calculates chemotaxis steering signals based on the bilateral odor intensity.

- **Observation keys used:** `"odor_intensity"`
- **Args:**
  - `obs` (`Observation`): Observation data.
- **Returns:**
  - `signal` (`np.ndarray`): Steering control signal for left and right motor systems.

---

### `compute_avoidance_steer(self, features)`
Generates a turning signal to steer away from visual obstacles based on feature map.

- **Args:**
  - `features` (`np.ndarray`): Visual features from `process_visual_obs`.
- **Returns:**
  - `steer` (`np.ndarray`): Left/right turning signal.

---

### `compute_ball_signal(self, obs)`
Detects a red object in raw RGB vision and returns a repulsive steering signal.

- **Observation keys used:** `"raw_vision"`
- **Args:**
  - `obs` (`Observation`): Observation data.
- **Returns:**
  - `red_seen` (`bool`): Whether a red object is seen.
  - `signal` (`np.ndarray`): Repulsion signal.

---

### `set_returning(self, flag: bool)`
Sets the controller to enter or exit homing mode.

- **Args:**
  - `flag` (`bool`): `True` to return to the nest, `False` to explore.

---

### `integrate_velocity(self, velocity, heading)`
Updates the internal estimate of the fly’s position using velocity and heading.

- **Args:**
  - `velocity` (`np.ndarray`): Velocity in body-centric coordinates.
  - `heading` (`float`): Global heading in radians.

---

### `compute_nest(self, position, theta, smooth_heading)`
Calculates the distance and angular offset between the fly and the nest.

- **Args:**
  - `position` (`np.ndarray`): Current position estimate.
  - `theta` (`float`): Angle to the nest.
  - `smooth_heading` (`float`): Smoothed heading.

---

### `step_back(self)`
Returns a fixed backward command used to unstuck the fly.

- **Returns:**
  - `signal` (`np.ndarray`): `[-1.0, -1.0]`

---

### `get_actions(self, obs: Observation) -> Action`
Main control logic that determines motor output based on sensory inputs and internal states.

- **Observation keys used:**
  - `"velocity"`
  - `"heading"`
  - `"odor_intensity"`
  - `"vision"`
  - `"raw_vision"`
  - `"reached_odour"`

- **Behavioral logic:**
  - If the fly has **not yet reached the odor source**:
    - If stuck: perform backward steps.
    - If in cooldown: use chemotaxis and ball detection to steer.
    - Otherwise: use chemotaxis and obstacle avoidance.
  - If the fly **has reached the odor source**:
    - Use path integration to compute direction back to the nest.
    - Navigate toward the nest until close enough to stop.

- **Args:**
  - `obs` (`Observation`): Current observation data.

- **Returns:**
  - `action` (`dict`): Dictionary with `"joints"` and `"adhesion"` values for the simulator.

---
