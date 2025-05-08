from typing import Callable
import cv2
import numpy as np
import os

from flygym.arena import FlatTerrain
from flygym.arena import BaseArena


def _get_random_target_position(
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    rng: np.random.Generator,
):
    """Generate a random target position.

    Parameters
    ----------
    distance_range : tuple[float, float]
        Distance range from the origin.
    angle_range : tuple[float, float]
        Angle rabge in radians.
    rng : np.random.Generator
        The random number generator.
    Returns
    -------
    np.ndarray
        The target position in the form of [x, y].
    """
    p = rng.uniform(*distance_range) * np.exp(1j * rng.uniform(*angle_range))
    return np.array([p.real, p.imag], float)


def _circ(
    img: np.ndarray,
    xy: tuple[float, float],
    r: float,
    value: bool,
    xmin: float,
    ymin: float,
    res: float,
    outer=False,
):
    """Draw a circle on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on.
    xy : tuple[float, float]
        The center of the circle.
    r : float
        The radius of the circle.
    value : bool
        The value to set the pixels to.
    xmin : float
        The minimum x value of the grid.
    ymin : float
        The minimum y value of the grid.
    res : float
        The resolution of the grid.
    outer : bool, optional
        If True, draw the outer circle. Otherwise, draw a filled circle.

    Returns
    -------
    None
    """
    center = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    radius = int(r / res) + 1 if outer else int(r / res)
    color = bool(value)
    thickness = 1 if outer else -1
    cv2.circle(img, center, radius, color, thickness)

class ObstacleOdorArena(BaseArena):
    num_sensors = 4

    def __init__(
        self,
        terrain: BaseArena,
        obstacle_positions: np.ndarray = np.array([(7.5, 0), (12.5, 5), (17.5, -5)]),
        obstacle_colors: np.ndarray | tuple = (0, 0, 0, 1),
        obstacle_radius: float = 1,
        obstacle_height: float = 4,
        odor_source: np.ndarray = np.array([[25, 0, 2]]),
        peak_odor_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: list[tuple[float, float, float, float]] | None = None,
        marker_size: float = 0.1,
    ):
        self.terrain_arena = terrain
        self.obstacle_positions = obstacle_positions
        self.root_element = terrain.root_element
        self.friction = terrain.friction
        self.obstacle_radius = obstacle_radius
        z_offset = terrain.get_spawn_position(np.zeros(3), np.zeros(3))[0][2]
        obstacle_colors = np.array(obstacle_colors)
        if obstacle_colors.shape == (4,):
            obstacle_colors = np.array(
                [obstacle_colors for _ in range(obstacle_positions.shape[0])]
            )
        else:
            assert obstacle_colors.shape == (obstacle_positions.shape[0], 4)

        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_odor_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.odor_dim = self.peak_odor_intensity.shape[1]
        self.diffuse_func = diffuse_func

        # Add markers at the odor sources
        if marker_colors is None:
            rgb = np.array([255, 127, 14]) / 255
            marker_colors = [(*rgb, 1)] * self.num_odor_sources
        self.marker_colors = marker_colors
        self._odor_marker_geoms = []
        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            pos = list(pos)
            pos[2] += z_offset
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            geom = marker_body.add(
                "geom",
                name="odor_source_marker_sensor",  # if we add sensor to the name it doesn't get added to the contact pairs: see `init_floor_contacts` in fly
                type="capsule",
                size=(marker_size, marker_size),
                rgba=rgba,
                contype=0,  # we also need to add these to not have it involved in any collisions
                conaffinity=0,
            )
            self._odor_marker_geoms.append(geom)

        # Reshape odor source and peak intensity arrays to simplify future calculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

        # Add obstacles
        self.obstacle_bodies = []
        obstacle_material = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.obstacle_z_pos = z_offset + obstacle_height / 2
        for i in range(obstacle_positions.shape[0]):
            obstacle_pos = [*obstacle_positions[i, :], self.obstacle_z_pos]
            obstacle_color = obstacle_colors[i]
            obstacle_body = self.root_element.worldbody.add(
                "body", name=f"obstacle_{i}", mocap=True, pos=obstacle_pos
            )
            self.obstacle_bodies.append(obstacle_body)
            obstacle_body.add(
                "geom",
                type="cylinder",
                size=(obstacle_radius, obstacle_height / 2),
                rgba=obstacle_color,
                material=obstacle_material,
                contype="0",
                conaffinity="8",
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.terrain_arena.get_spawn_position(rel_pos, rel_angle)

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    def pre_visual_render_hook(self, physics):
        # hide elements that we don't want to see in the fly's vision
        # but only in the external camera
        for geom, rgba in zip(
            self._odor_marker_geoms,
            self.marker_colors,
        ):
            physics.bind(geom).rgba = np.array([*rgba[:3], 0])
        ground = self.terrain_arena.root_element.find(
            namespace="geom", identifier="ground"
        )
        assert ground is not None, "Couldn't find the ground"
        physics.bind(ground).rgba = np.array([*physics.bind(ground).rgba[:3], 0])

    def post_visual_render_hook(self, physics):
        # show elements that we don't want to see in the fly's vision
        # but only in the external camera
        for geom, rgba in zip(
            self._odor_marker_geoms,
            self.marker_colors,
        ):
            physics.bind(geom).rgba = np.array([*rgba[:3], 1])
        ground = self.terrain_arena.root_element.find(
            namespace="geom", identifier="ground"
        )
        assert ground is not None, "Couldn't find the ground"
        physics.bind(ground).rgba = np.array([*physics.bind(ground).rgba[:3], 1])

    def _get_max_floor_height(self) -> float:
        return self.terrain_arena._get_max_floor_height()


class OdorTargetOnlyArena(ObstacleOdorArena):
    def __init__(
        self,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        to_target_distance=3.0,
        seed=None,
        **kwargs,
    ):
        self.fly = fly
        self.quit = False
        self.to_target_distance = to_target_distance

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.target_position = _get_random_target_position(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            rng=self.rng,
        )

        super().__init__(
            terrain=FlatTerrain(ground_alpha=1),
            obstacle_positions=np.array([]),
            peak_odor_intensity=np.array([[1, 0]]),
            odor_source=np.array([[*self.target_position, 2]]),
            marker_colors=np.array([target_marker_color]),
            marker_size=target_marker_size,
            **kwargs,
        )

    def step(self, dt, physics):
        fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
        if np.linalg.norm(self.target_position - fly_pos) < self.to_target_distance:
            self.quit = True

    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)


class ScatteredPillarsArena(ObstacleOdorArena):
    """
    An arena with scattered pillars and a target marker.

    This class generates an arena with randomly placed pillars and a target marker.
    The target marker is placed at a random position within a specified distance
    and angle range. Pillars are placed randomly while maintaining a minimum
    separation from the target, the fly, and other pillars.

    Parameters
    ----------
    target_distance_range : tuple[float, float], optional
        Range of distances from the origin for the target marker. Default is (29, 31).
    target_angle_range : tuple[float, float], optional
        Range of angles (in radians) for the target marker. Default is (-π, π).
    target_clearance_radius : float, optional
        Minimum clearance radius around the target marker. Default is 4.
    target_marker_size : float, optional
        Size of the target marker. Default is 0.3.
    target_marker_color : tuple[float, float, float, float], optional
        RGBA color of the target marker. Default is (1, 0.5, 14/255, 1).
    pillar_height : float, optional
        Height of the pillars. Default is 3.
    pillar_radius : float, optional
        Radius of the pillars. Default is 0.3.
    pillars_minimum_separation : float, optional
        Minimum separation between pillars. Default is 6.
    fly_clearance_radius : float, optional
        Minimum clearance radius around the fly. Default is 4.
    seed : int or None, optional
        Seed for the random number generator. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    odor_source : np.ndarray
        Position of the odor source.
    marker_colors : np.ndarray
        Colors of the markers.
    peak_odor_intensity : np.ndarray
        Peak intensity of the odor.
    obstacle_positions : np.ndarray
        Positions of the pillars.
    obstacle_radius : float
        Radius of the pillars.
    obstacle_height : float
        Height of the pillars.
    terrain : FlatTerrain
        Terrain of the arena.
    marker_size : float
        Size of the target marker.
    """

    def __init__(
        self,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=3,
        pillar_radius=0.3,
        pillars_minimum_separation=6,
        fly_clearance_radius=4,
        seed=None,
        to_target_distance=3.0,
        **kwargs,
    ):
        self.fly = fly
        self.quit = False
        self.to_target_distance = to_target_distance

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.target_position = _get_random_target_position(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            rng=self.rng,
        )

        pillar_positions = self._get_pillar_positions(
            target_position=self.target_position,
            target_clearance_radius=target_clearance_radius,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            rng=self.rng,
        )

        ObstacleOdorArena.__init__(
            self,
            terrain=FlatTerrain(ground_alpha=1),
            obstacle_positions=pillar_positions,
            obstacle_radius=pillar_radius,
            obstacle_height=pillar_height,
            odor_source=np.array([[*self.target_position, 1]]),
            peak_odor_intensity=np.array([[1, 0]]),
            marker_colors=np.array([target_marker_color]),
            marker_size=target_marker_size,
            **kwargs,
        )

    @staticmethod
    def _get_pillar_positions(
        target_position: tuple[float, float],
        target_clearance_radius: float,
        pillar_radius: float,
        pillars_minimum_separation: float,
        fly_clearance_radius: float,
        rng: np.random.Generator,
        res: float = 0.05,
    ):
        """Generate random pillar positions.

        Parameters
        ----------
        target_position : tuple[float, float]
            The target x and y coordinates.
        target_clearance_radius : float
            The radius of the area around the target that should be clear of pillars.
        pillar_radius : float
            The radius of the pillars.
        pillars_minimum_separation : float
            Minimum separation between pillars.
        fly_clearance_radius : float
            The radius of the area around the fly that should be clear of pillars.
        rng : np.random.Generator
            The random number generator.
        res : float, optional
            The resolution of the grid. Default is 0.05.

        Returns
        -------
        np.ndarray
            The positions of the pillars in the form of [[x1, y1], [x2, y2], ...].
        """
        pillar_clearance_radius = pillar_radius * 2 + pillars_minimum_separation
        target_clearance_radius = target_clearance_radius + pillar_radius
        fly_clearance_radius = fly_clearance_radius + pillar_radius

        target_position = np.asarray(target_position)
        distance = np.linalg.norm(target_position)
        xmin = ymin = -distance
        xmax = ymax = distance
        n_cols = int((xmax - xmin) / res)
        n_rows = int((ymax - ymin) / res)
        im1 = np.zeros((n_rows, n_cols), dtype=np.uint8)
        im2 = np.zeros((n_rows, n_cols), dtype=np.uint8)

        _circ(im1, (0, 0), distance, 1, xmin, ymin, res)
        _circ(im1, (0, 0), fly_clearance_radius, 0, xmin, ymin, res)
        _circ(im1, target_position, target_clearance_radius, 0, xmin, ymin, res)

        pillars_xy = [target_position / 2]
        _circ(im1, pillars_xy[0], pillar_clearance_radius, 0, xmin, ymin, res)
        _circ(
            im2, pillars_xy[0], pillar_clearance_radius, 1, xmin, ymin, res, outer=True
        )

        while True:
            argwhere = np.argwhere(im1 & im2)
            try:
                p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
            except ValueError:
                break
            pillars_xy.append(p)
            _circ(im1, p, pillar_clearance_radius, 0, xmin, ymin, res)
            _circ(im2, p, pillar_clearance_radius, 1, xmin, ymin, res, outer=True)

        return np.array(pillars_xy)

    def step(self, dt, physics):
        fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
        if np.linalg.norm(self.target_position - fly_pos) < self.to_target_distance:
            self.quit = True

    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)


class LoomingBallArena(OdorTargetOnlyArena):
    """
    Simulates a looming ball scenario where a ball approaches a fly entity from different angles
    with Poisson-distributed spawning events.
    """

    def __init__(
        self,
        timestep,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        to_target_distance=3.0,
        ball_radius=1.0,
        ball_approach_vel=50,
        ball_approach_start_radius=20,
        ball_overshoot_dist=5,
        looming_lambda=1.0,
        seed=None,
        approach_angles=np.array([np.pi / 4, 3 * np.pi / 4]),
        **kwargs,
    ):
        OdorTargetOnlyArena.__init__(
            self,
            fly=fly,
            target_distance_range=target_distance_range,
            target_angle_range=target_angle_range,
            target_marker_size=target_marker_size,
            target_marker_color=target_marker_color,
            to_target_distance=to_target_distance,
            seed=seed,
            **kwargs,
        )

        self.ball_specific_init(
            timestep=timestep,
            ball_radius=ball_radius,
            ball_approach_vel=ball_approach_vel,
            ball_approach_start_radius=ball_approach_start_radius,
            ball_overshoot_dist=ball_overshoot_dist,
            looming_lambda=looming_lambda,
            approach_angles=approach_angles,
        )

    def ball_specific_init(
        self,
        timestep,
        ball_radius,
        ball_approach_vel,
        ball_approach_start_radius,
        ball_overshoot_dist,
        looming_lambda,
        approach_angles,
    ):
        self.dt = timestep
        self.ball_radius = ball_radius

        self._setup_probabilities(looming_lambda)
        self._setup_trajectory_params(
            ball_approach_vel, ball_approach_start_radius, ball_overshoot_dist
        )
        self._setup_ball_heights()

        self.ball_approach_angles = approach_angles
        self.is_looming = False
        self.ball_traj_advancement = 0

        self._setup_velocity_buffer()
        self.add_ball(ball_radius)

    def _setup_probabilities(self, looming_lambda):
        """Initialize Poisson probability settings."""
        self.p_no_looming = np.exp(-looming_lambda * self.dt)

    def _setup_trajectory_params(self, vel, start_radius, overshoot_dist):
        """Precompute trajectory parameters for the looming ball."""
        self.ball_approach_vel = vel
        self.ball_approach_start_radius = start_radius
        self.overshoot_dist = overshoot_dist

        interception_time = start_radius / vel
        self.n_interception_steps = int(interception_time / self.dt)
        self.n_overshoot_steps = int((overshoot_dist / vel) / self.dt)

        self.ball_trajectory = np.zeros(
            (self.n_interception_steps + self.n_overshoot_steps, 2)
        )

    def _setup_ball_heights(self):
        """Calculate ball positions for visible and resting states."""
        self.ball_rest_height = 10.0
        self.ball_act_height = self.ball_radius/4*3 + self._get_max_floor_height() 

    def _setup_velocity_buffer(self):
        """Setup a fixed-length FIFO buffer for fly velocity estimation."""
        self.vel_buffer_size = 500
        self.fly_velocities = np.full((self.vel_buffer_size, 2), np.nan)
        self.fly_velocities_idx = 0

    def add_ball(self, ball_radius):
        """Add the ball to the scene with joints and geometry."""
        self.ball_body = self.root_element.worldbody.add(
            "body",
            name="ball_mocap",
            pos=[0, 0, self.ball_rest_height],
            mocap=True,
        )

        self.ball_geom = self.ball_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=[ball_radius],
            rgba=[1, 0, 0, 0],
            density=1.0,
        )

    def set_ball_trajectory(self, start_pts, end_pts):
        """Generate a linear trajectory from start to end."""
        self.ball_trajectory = np.linspace(
            start_pts, end_pts, self.n_interception_steps + self.n_overshoot_steps
        )

    def make_ball_visible(self, physics):
        physics.bind(self.ball_geom).rgba[3] = 1

    def make_ball_invisible(self, physics):
        physics.bind(self.ball_geom).rgba[3] = 0

    def move_ball(self, physics, x, y, z):
        """Move ball to the desired location using joint positions."""
        physics.bind(self.ball_body).mocap_pos = np.array([x, y, z])

    def _get_mean_fly_velocity(self):
        """Compute average fly velocity from buffer."""
        return np.nanmean(self.fly_velocities, axis=0)

    def _should_trigger_ball(self):
        """Check if ball should start looming based on Poisson process."""
        return self.rng.uniform() > self.p_no_looming and not self.is_looming

    def _compute_trajectory_from_fly(self, fly_pos, fly_vel, fly_or_vec):
        """Generate start/end points of the ball trajectory based on fly state."""
        fly_roll = np.arctan2(fly_or_vec[1], fly_or_vec[0])
        approach_side = self.rng.choice([-1, 1])
        rel_angles = self.ball_approach_angles * approach_side + fly_roll
        if approach_side == -1:
            rel_angles = rel_angles[::-1]
        start_angle = self.rng.uniform(low=rel_angles[0], high=rel_angles[1])

        interception_pos = fly_pos + fly_vel * self.n_interception_steps * self.dt
        start_pos = interception_pos + self.ball_approach_start_radius * np.array(
            [np.cos(start_angle), np.sin(start_angle)]
        )
        end_pos = interception_pos - self.overshoot_dist * np.array(
            [np.cos(start_angle), np.sin(start_angle)]
        )
        return start_pos, end_pos, interception_pos, start_angle

    def step_ball(self, dt, physics):
        """Advance the ball along its trajectory."""
        # Update fly velocity buffer
        fly_vel = physics.bind(self.fly._body_sensors[1]).sensordata[:2].copy()
        self.fly_velocities[self.fly_velocities_idx % self.vel_buffer_size] = fly_vel
        self.fly_velocities_idx += 1

        if self._should_trigger_ball():
            self.is_looming = True
            self.ball_traj_advancement = 0
            self.make_ball_visible(physics)

            fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
            fly_or_vec = physics.bind(self.fly._body_sensors[4]).sensordata.copy()
            fly_vel_mean = self._get_mean_fly_velocity()

            (
                start_pts,
                end_pts,
                interception_pos,
                angle,
            ) = self._compute_trajectory_from_fly(fly_pos, fly_vel_mean, fly_or_vec)

            self.set_ball_trajectory(start_pts, end_pts)
            self.move_ball(physics, *start_pts, self.ball_act_height)
            self.ball_traj_advancement += 1

            # Optional: visualize
            # self._plot_trajectory_debug(fly_pos, fly_vel_mean, interception_pos, start_pts, fly_or_vec)

        elif self.is_looming:
            self._advance_ball(physics)
        else:
            self.move_ball(physics, 0, 0, self.ball_rest_height)

    def step(self, dt, physics):
        """Main loop: updates ball state, triggers looming events, and moves the ball."""

        OdorTargetOnlyArena.step(self, dt, physics)
        self.step_ball(dt, physics)

    def _advance_ball(self, physics):
        """Advance the ball along its trajectory."""
        pos = self.ball_trajectory[self.ball_traj_advancement]
        self.move_ball(physics, pos[0], pos[1], self.ball_act_height)
        self.ball_traj_advancement += 1

        if (
            self.ball_traj_advancement
            >= self.n_interception_steps + self.n_overshoot_steps
        ):
            self.is_looming = False
            self.make_ball_invisible(physics)
            self.move_ball(physics, 0, 0, self.ball_rest_height)

    def _plot_trajectory_debug(
        self, fly_pos, fly_vel, intercept_pos, start_pos, orientation_vec
    ):
        """Visualize trajectory for debugging."""
        plt.scatter(fly_pos[0], fly_pos[1], label="fly pos", s=5)
        plt.scatter(
            intercept_pos[0], intercept_pos[1], label="fly interception pos", s=5
        )
        plt.plot(
            self.ball_trajectory[:, 0],
            self.ball_trajectory[:, 1],
            label="ball trajectory",
        )
        plt.scatter(start_pos[0], start_pos[1], label="ball start pos", s=5)
        plt.arrow(
            fly_pos[0], fly_pos[1], fly_vel[0], fly_vel[1], head_width=0.5, fc="blue"
        )
        plt.arrow(
            intercept_pos[0],
            intercept_pos[1],
            orientation_vec[0],
            orientation_vec[1],
            head_width=0.5,
            fc="green",
        )
        plt.legend()
        plt.show()

    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.is_looming = False
        self.ball_traj_advancement = 0
        self.make_ball_invisible(physics)
        self.ball_trajectory = np.zeros((self.n_interception_steps, 2))
        self.move_ball(physics, 0, 0, self.ball_rest_height)


class HierarchicalArena(ScatteredPillarsArena, LoomingBallArena):
    """
    A hierarchical arena that combines the features of both OdorTargetOnlyArena and ScatteredPillarsArena.
    This class allows for a more complex environment with both target and scattered pillars.
    """

    def __init__(
        self,
        timestep,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=3,
        pillar_radius=0.3,
        pillars_minimum_separation=6,
        fly_clearance_radius=4,
        seed=None,
        to_target_distance=3.0,
        ball_radius=1.0,
        ball_approach_vel=50,
        ball_approach_start_radius=20,
        ball_overshoot_dist=5,
        looming_lambda=1.0,
        approach_angles=np.array([np.pi / 4, 3 * np.pi / 4]),
        **kwargs,
    ):
        ScatteredPillarsArena.__init__(
            self,
            fly=fly,
            target_distance_range=target_distance_range,
            target_angle_range=target_angle_range,
            target_clearance_radius=target_clearance_radius,
            target_marker_size=target_marker_size,
            target_marker_color=target_marker_color,
            pillar_height=pillar_height,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            seed=seed,
            **kwargs,
        )

        self.ball_specific_init(
            timestep=timestep,
            ball_radius=ball_radius,
            ball_approach_vel=ball_approach_vel,
            ball_approach_start_radius=ball_approach_start_radius,
            ball_overshoot_dist=ball_overshoot_dist,
            looming_lambda=looming_lambda,
            approach_angles=approach_angles,
        )

    def step(self, dt, physics):
        """Update the arena state and check for target acquisition."""
        LoomingBallArena.step(self, dt, physics)


class FoodToNestArena(HierarchicalArena):
    def __init__(
        self,
        timestep,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=3,
        pillar_radius=0.3,
        pillars_minimum_separation=6,
        fly_clearance_radius=4,
        seed=None,
        to_target_distance=3.0,
        ball_radius=1.0,
        ball_approach_vel=50,
        ball_approach_start_radius=20,
        ball_overshoot_dist=5,
        looming_lambda=1.0,
        approach_angles=np.array([np.pi / 4, 3 * np.pi / 4]),
        **kwargs,
    ):
        HierarchicalArena.__init__(
            self,
            timestep=timestep,
            fly=fly,
            target_distance_range=target_distance_range,
            target_angle_range=target_angle_range,
            target_clearance_radius=target_clearance_radius,
            target_marker_size=target_marker_size,
            target_marker_color=target_marker_color,
            pillar_height=pillar_height,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            seed=seed,
            to_target_distance=to_target_distance,
            ball_radius=ball_radius,
            ball_approach_vel=ball_approach_vel,
            ball_approach_start_radius=ball_approach_start_radius,
            ball_overshoot_dist=ball_overshoot_dist,
            looming_lambda=looming_lambda,
            approach_angles=approach_angles,
        )

        self.state = "exploration"  # or "returning"

        self.nest_position = np.array([0, 0])
        # add nest indicator
        self.nest_marker = self.root_element.worldbody.add(
            "body",
            pos=[0, 0, 2],
        )
        torus_path = os.path.dirname(__file__) + "/../assets/cone.stl"
        self.root_element.asset.add(
            "mesh", name="nest_mesh", file=torus_path, scale=[1, 1, 1]
        )
        self.nest_geom = self.nest_marker.add(
            "geom", name="nest", type="mesh", mesh="nest_mesh", rgba=[0, 1, 0, 0]
        )

        self.pillar_height = pillar_height

    def pre_visual_render_hook(self, physics):
        OdorTargetOnlyArena.pre_visual_render_hook(self, physics)
        physics.bind(self.nest_geom).rgba[3] = 0

    def post_visual_render_hook(self, physics):
        OdorTargetOnlyArena.post_visual_render_hook(self, physics)
        physics.bind(self.nest_geom).rgba[3] = 1

    def setup_return_mode(self, physics):
        physics.bind(self.nest_geom).rgba[3] = 1
        for geom in self._odor_marker_geoms:
            physics.bind(geom).rgba[3] = 0

        # move the pillars under the arena and hide them
        for body in self.obstacle_bodies:
            physics.bind(body).mocap_pos[2] = -self.pillar_height - 5.0
            physics.bind(body.find_all("geom")[0]).rgba = np.array([0, 0, 0, 0])

        self.move_ball(physics, 0, 0, self.ball_rest_height)
        self.make_ball_invisible(physics)

    def setup_exploration_mode(self, physics):
        physics.bind(self.nest_geom).rgba[3] = 0
        for geom in self._odor_marker_geoms:
            physics.bind(geom).rgba[3] = 1
        for body in self.obstacle_bodies:
            physics.bind(body).mocap_pos[2] = self.pillar_height

    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        HierarchicalArena.reset(self, physics, seed)
        self.state = "exploration"
        self.setup_exploration_mode(physics)

    def step(self, dt, physics):
        fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
        if self.state == "exploration":
            LoomingBallArena.step_ball(self, dt, physics)
            if np.linalg.norm(self.target_position - fly_pos) < self.to_target_distance:
                self.state = "returning"
                self.setup_return_mode(physics)
                
class FoodToNestArenaWithoutObstacles(HierarchicalArena):
    def __init__(
        self,
        timestep,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=1e-10,
        pillar_radius=0.3,
        pillars_minimum_separation=1e6,
        fly_clearance_radius=4,
        seed=None,
        to_target_distance=3.0,
        ball_radius=1e-10,
        ball_approach_vel=50,
        ball_approach_start_radius=20,
        ball_overshoot_dist=5,
        looming_lambda=1.0,
        approach_angles=np.array([np.pi / 4, 3 * np.pi / 4]),
        **kwargs,
    ):
        HierarchicalArena.__init__(
            self,
            timestep=timestep,
            fly=fly,
            target_distance_range=target_distance_range,
            target_angle_range=target_angle_range,
            target_clearance_radius=target_clearance_radius,
            target_marker_size=target_marker_size,
            target_marker_color=target_marker_color,
            pillar_height=pillar_height,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            seed=seed,
            to_target_distance=to_target_distance,
            ball_radius=ball_radius,
            ball_approach_vel=ball_approach_vel,
            ball_approach_start_radius=ball_approach_start_radius,
            ball_overshoot_dist=ball_overshoot_dist,
            looming_lambda=looming_lambda,
            approach_angles=approach_angles,
        )

        self.state = "exploration"  # or "returning"

        self.nest_position = np.array([0, 0])
        # add nest indicator
        self.nest_marker = self.root_element.worldbody.add(
            "body",
            pos=[0, 0, 2],
        )
        torus_path = os.path.dirname(__file__) + "/../assets/cone.stl"
        self.root_element.asset.add(
            "mesh", name="nest_mesh", file=torus_path, scale=[1, 1, 1]
        )
        self.nest_geom = self.nest_marker.add(
            "geom", name="nest", type="mesh", mesh="nest_mesh", rgba=[0, 1, 0, 0]
        )

        self.pillar_height = 0
        

    def pre_visual_render_hook(self, physics):
        OdorTargetOnlyArena.pre_visual_render_hook(self, physics)
        physics.bind(self.nest_geom).rgba[3] = 0

    def post_visual_render_hook(self, physics):
        OdorTargetOnlyArena.post_visual_render_hook(self, physics)
        physics.bind(self.nest_geom).rgba[3] = 1

    def setup_return_mode(self, physics):
        physics.bind(self.nest_geom).rgba[3] = 1
        for geom in self._odor_marker_geoms:
            physics.bind(geom).rgba[3] = 0

        # move the pillars under the arena and hide them
        for body in self.obstacle_bodies:
            physics.bind(body).mocap_pos[2] = -self.pillar_height - 5.0
            physics.bind(body.find_all("geom")[0]).rgba = np.array([0, 0, 0, 0])

        self.move_ball(physics, 0, 0, self.ball_rest_height)
        self.make_ball_invisible(physics)

    def setup_exploration_mode(self, physics):
        physics.bind(self.nest_geom).rgba[3] = 0
        for geom in self._odor_marker_geoms:
            physics.bind(geom).rgba[3] = 1
        for body in self.obstacle_bodies:
            # physics.bind(body).mocap_pos[2] = self.pillar_height
            physics.bind(body).mocap_pos[2] = -self.pillar_height - 5.0
            physics.bind(body.find_all("geom")[0]).rgba = np.array([0, 0, 0, 0])
    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        
        HierarchicalArena.reset(self, physics, seed)
        self.state = "exploration"
        self.setup_exploration_mode(physics)

    def step(self, dt, physics):
        fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
        if self.state == "exploration":
            LoomingBallArena.step_ball(self, dt, physics)
            if np.linalg.norm(self.target_position - fly_pos) < self.to_target_distance:
                self.state = "returning"
                self.setup_return_mode(physics)