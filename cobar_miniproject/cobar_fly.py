import collections
from flygym import Fly, Simulation
import numpy as np
# I tried to JIT this with numba but it was slower
def quat_to_zyx(q0, q1, q2, q3):
    """
    Optimised code for converting a quaternion to ZYX angles.

    `quat_to_zyx(*quat)` is equivalent to `dm_control.utils.transformations.quat_to_euler(quat, ordering="ZYX")`
    """
    q22 = q2 * q2
    rmat_0_0 = 1 / 2 - q22 - q3 * q3
    rmat_1_0 = q1 * q2 + q3 * q0
    rmat_2_0 = 2 * (q1 * q3 - q2 * q0)
    rmat_2_1 = q2 * q3 + q1 * q0
    rmat_2_2 = 1 / 2 - q1 * q1 - q22

    x = np.arctan2(rmat_2_1, rmat_2_2)
    y = -np.arcsin(rmat_2_0)
    z = np.arctan2(rmat_1_0, rmat_0_0)

    return np.array([z, y, x])

class CobarFly(Fly):
    def __init__(
        self, debug=False, enable_vision=True, render_raw_vision=True, **kwargs
    ):
        """Specific Fly instance for use in the COBAR project.
        The physics have been to improve simulation speed.

        Args:
            debug (bool, optional): If this is true, get the raw observations from the fly. Don't use this when running your controller! Defaults to False.
            enable_vision (bool, optional): Whether the fly-perspective vision is rendered. This can be disabled if not needed to increase simulation speed. Defaults to True.
            render_raw_vision (bool, optional): Whether the raw fly-perspective vision is rendered. This can be disabled if not needed to increase simulation speed. Defaults to False.
        """
        super().__init__(
            contact_sensor_placements=[
                f"{side}{pos}{segment}"
                for side in "LR"
                for pos in "FMH"
                for segment in ["Tarsus5"]
            ],
            self_collisions="none",
            floor_collisions=[
                f"{side}{pos}{dof}"
                for side in "LR"
                for pos in "FMH"
                for dof in ["Tarsus5"]
            ],
            xml_variant="seqik_simple",
            head_stabilization_model="thorax",
            neck_kp=1000,
            enable_adhesion=True,
            enable_olfaction=True,
            enable_vision=enable_vision,
            render_raw_vision=render_raw_vision,
            vision_refresh_rate=100,
            **kwargs,
        )
        self.debug = debug

        for geom in self.model.find_all("geom"):
            if geom.name[2:] in {"Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"}:
                geom.contype = "8"
            elif geom.name in {"Head", "Thorax", "A1A2", "A3", "A4", "A5"}:
                geom.contype = "9"

        # add these joints to enable using head stabilisation models
        fly_head_body = self.model.worldbody.find("body", "Head")  # type: ignore
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_yaw",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "1 0 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 1 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_roll",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 0 1",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        self.model.visual.map.zfar = 10  # type: ignore

        self._geoms_to_hide = [
            "A1A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "Haustellum",
            "Head",
            "LArista",
            "LEye",
            "LFCoxa",
            "LFFemur",
            "LFTarsus1",
            "LFTarsus2",
            "LFTarsus3",
            "LFTarsus4",
            "LFTarsus5",
            "LFTibia",
            "LFuniculus",
            "LHCoxa",
            "LHFemur",
            "LHTarsus1",
            "LHTarsus2",
            "LHTarsus3",
            "LHTarsus4",
            "LHTarsus5",
            "LHTibia",
            "LHaltere",
            "LMCoxa",
            "LMFemur",
            "LMTarsus1",
            "LMTarsus2",
            "LMTarsus3",
            "LMTarsus4",
            "LMTarsus5",
            "LMTibia",
            "LPedicel",
            "LWing",
            "RArista",
            "REye",
            "RFCoxa",
            "RFFemur",
            "RFTarsus1",
            "RFTarsus2",
            "RFTarsus3",
            "RFTarsus4",
            "RFTarsus5",
            "RFTibia",
            "RFuniculus",
            "RHCoxa",
            "RHFemur",
            "RHTarsus1",
            "RHTarsus2",
            "RHTarsus3",
            "RHTarsus4",
            "RHTarsus5",
            "RHTibia",
            "RHaltere",
            "RMCoxa",
            "RMFemur",
            "RMTarsus1",
            "RMTarsus2",
            "RMTarsus3",
            "RMTarsus4",
            "RMTarsus5",
            "RMTibia",
            "RPedicel",
            "RWing",
            "Rostrum",
            "Thorax",
        ]
    
    def reset(self, sim, **kwargs):
        obs, info = super().reset(sim, **kwargs)
        obs["vision_updated"] = True
        obs["reached_odour"] = False
        return obs, info

    def get_observation(self, sim: Simulation):
        # if we're running the fly in debug mode (for development) it will return all raw observations
        # otherwise, it will return only a reduced observation space with egocentric observations
        if self.debug:
            raw_obs = super().get_observation(sim)

        physics = sim.physics
        actuated_joint_sensordata = physics.bind(
            self._actuated_joint_sensors
        ).sensordata

        joint_obs = np.array(actuated_joint_sensordata).reshape((3, -1), order="F")
        joint_obs[2, :] *= 1e-9  # convert to N
        joint_obs = joint_obs[:, self._monitored_joint_order]

        if len(self.monitored_joints) > len(self.actuated_joints):
            raise NotImplementedError(
                "Cobar fly shouldn't have any non actuated joints"
            )

        # fly position and orientation
        fly_pos = physics.bind(self._body_sensors[0]).sensordata
        fly_vel = physics.bind(self._body_sensors[1]).sensordata
        quat = physics.bind(self._body_sensors[2]).sensordata

        ang_pos = quat_to_zyx(*quat)

        # needed for the camera to track the fly
        self.last_obs["rot"] = ang_pos
        self.last_obs["pos"] = fly_pos

        # contact forces from crf_ext (first three components are rotational)
        contact_forces = physics.named.data.cfrc_ext[self.contact_sensor_placements][
            :, 3:
        ].copy()
        if self.enable_adhesion:
            # Adhesion adds force to the contact. Let's compute this force
            # and remove it from the contact forces
            contactid_normal = collections.defaultdict(list)

            geoms_we_care_about_to_sensor_ids = {
                geom: sensor_id
                for geom, sensor_id, last_adhesion in zip(
                    self._adhesion_actuator_geom_id,
                    self._adhesion_bodies_with_contact_sensors,
                    self._last_adhesion,
                )
                if last_adhesion
            }
            geoms_we_care_about = set(geoms_we_care_about_to_sensor_ids.keys())

            for contact in physics.data.contact:
                if contact.exclude == 0:
                    if contact.geom1 in geoms_we_care_about:
                        contactid_normal[
                            geoms_we_care_about_to_sensor_ids[contact.geom1]
                        ].append(contact.frame[:3])
                    if contact.geom2 in geoms_we_care_about:
                        contactid_normal[
                            geoms_we_care_about_to_sensor_ids[contact.geom2]
                        ].append(contact.frame[:3])

            for contact_sensor_id, normals in contactid_normal.items():
                # sum() / len() is the same as np.mean(normals, axis=0) but faster
                contact_forces[contact_sensor_id, :] -= (
                    self.adhesion_force * sum(normals) / len(normals)
                )

            # say which adhesions sensors are in contact with something
            self._active_adhesion = np.array(
                [
                    sensor_id in contactid_normal
                    for sensor_id in self._adhesion_bodies_with_contact_sensors
                ]
            )

        # end effector position
        ee_pos = physics.bind(self._end_effector_sensors).sensordata
        ee_pos = ee_pos.reshape((self.n_legs, 3))

        orientation_vec = physics.bind(self._body_sensors[4]).sensordata
        fly_angle = np.arctan2(*orientation_vec[1::-1])
        fly_pos_xy = fly_pos[:2].reshape(1, 2)

        end_effector_positions_relative = CobarFly.absolute_to_relative_pos(
            ee_pos[:, :2],
            fly_pos_xy,
            fly_angle,
        )
        velocities_relative = CobarFly.absolute_to_relative_pos(
            fly_vel[:2],
            np.zeros(2),
            fly_angle,
        )

        obs = {
            "joints": joint_obs.astype(np.float32),
            "end_effectors": end_effector_positions_relative.astype(np.float32),
            "contact_forces": contact_forces.astype(np.float32),
            "heading": fly_angle.astype(np.float32),
            "velocity": velocities_relative.astype(np.float32),
        }

        if self.enable_olfaction:
            antennae_pos = physics.bind(self._antennae_sensors).sensordata
            odor_intensity = sim.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity.astype(np.float32)

        if self.enable_vision:
            self._update_vision(sim)
            obs["vision"] = self._curr_visual_input
            if self.render_raw_vision:
                obs["raw_vision"] = self.get_info()["raw_vision"]

        # merge the observations
        if self.debug:
            for raw_key in raw_obs:
                obs["debug_" + raw_key] = raw_obs[raw_key]

        return obs

    def pre_step(self, action, sim: "Simulation"):
        physics = sim.physics
        joint_action = action["joints"]

        # estimate necessary neck actuation signals for head stabilization
        if self.head_stabilization_model == "thorax":
            # get roll and pitch from thorax rotation matrix (it seems to be transposed?)
            rmat_0_2, rmat_1_2, rmat_2_2 = physics.bind(self.thorax).xmat[6:]
            pitch = np.arcsin(rmat_0_2)
            roll = -np.arctan2(rmat_1_2, rmat_2_2)
            neck_actuation = np.array([roll, pitch])

            joint_action = np.concatenate((joint_action, neck_actuation))
            self._last_neck_actuation = neck_actuation
            physics.bind(self.actuators + self.neck_actuators).ctrl = joint_action
        else:
            raise NotImplementedError(
                "Cobar fly only supports thorax head stabilization."
            )

        if self.enable_adhesion:
            physics.bind(self.adhesion_actuators).ctrl = action["adhesion"]
            self._last_adhesion = action["adhesion"]

    def post_step(self, sim: Simulation):
        obs = self.get_observation(sim)
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        if self.enable_vision:
            vision_updated_this_step = sim.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        # Fly has flipped if the z component of the "up" cardinal vector is negative
        info["flip"] = sim.physics.bind(self._body_sensors[6]).sensordata[2] < 0

        if self.head_stabilization_model is not None:
            # this is tracked to decide neck actuation for the next step
            self._last_observation = obs
            info["neck_actuation"] = self._last_neck_actuation

        # add some extra fields to the obs so the controller can access them
        if self.enable_vision:
            obs["vision_updated"] = info["vision_updated"]
        obs["reached_odour"] = (
            getattr(sim.arena, "state", "exploring") == "returning"
        )  # this is only relevant for the path integration

        return obs, reward, terminated, truncated, info

    @staticmethod
    def absolute_to_relative_pos(
        pos: np.ndarray, base_pos: np.ndarray, heading_angle: float
    ) -> np.ndarray:
        """
        This function converts an absolute position to a relative position
        with respect to the base position and heading of the fly.

        It is used to obtain the fly-centric end effector (leg tip) positions.

        Args:
            pos (np.ndarray): Nx2 array of end effector positions in global coordinates.
            base_pos (np.ndarray): 1x2 array of the fly's x-y position.
            heading_angle (float): heading angle of the fly

        Returns:
            np.ndarray: Nx2 array of end effector positions in fly-centric coordinates
        """
        rel_pos = pos - base_pos
        rot_matrix = np.array(
            [
                [np.cos(-heading_angle), np.sin(-heading_angle)],
                [-np.sin(-heading_angle), np.cos(-heading_angle)],
            ]
        )
        return rel_pos @ rot_matrix
