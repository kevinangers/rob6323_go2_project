# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import torch
from collections.abc import Sequence
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device) # shape : (num_envs, action_dim)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # 12 is number of joints
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits

        # Actuator friction parameters
        self.mu_v = torch.zeros(self.num_envs, 12, device=self.device)
        self.F_s = torch.zeros(self.num_envs, 12, device=self.device)

        # Torque Regularization
        self.torques = torch.zeros(self.num_envs, 12, device=self.device)

        # Logging
        self._episode_sums = { # running sum of per step rewards for logging
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                "rew_torque",
                "foot2contact",
                "feet_air_time",
            ]
        }
        # variables needed for action rate penalization
        # Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids = []
        self._feet_ids_sensor = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            robot_ids, _ = self.robot.find_bodies(name)
            self._feet_ids.append(robot_ids[0])

            sensor_ids, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(sensor_ids[0])

        # Nominal stance foot height relative to base (used for feet clearance reward w/ uneven terrain)
        self._nominal_foot_z_rel = torch.zeros(self.num_envs, 1, device=self.device)

        # Variables needed for the raibert heuristic
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = (self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos) 

    def _apply_action(self) -> None:
        # Compute PD torques
        torques_pd = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        
        # Apply actuator friction model
        tau_stiction = self.F_s * torch.tanh(self.robot.data.joint_vel / 0.1)
        tau_viscous = self.mu_v * self.robot.data.joint_vel
        tau_friction = tau_stiction + tau_viscous

        torques = torch.clip(torques_pd - tau_friction, -self.torque_limits, self.torque_limits)
        self.torques = torques  # For torque regularization

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # -- Linear velocity tracking --
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1) # squared error between commanded [vx, vy] and actual [vx, vy]
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) # map to [0, 1] using exponential mapping (good tracking ~ 1, bad tracking ~ 0)
        
        # -- Yaw rate tracking --
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # -- Action rate penalization --
        # First derivative (Current - Last)
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
        # Second derivative (Current - 2*Last + 2ndLast)
        rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # -- Desired contact states from raibert heuristic --
        self._step_contact_targets()  
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # -- Penalize non-vertical orientation (projected gravity on XY plane) --
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)

        # -- Penalize vertical velocity (z-component of base linear velocity) --
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

        # -- Penalize high joint velocities --
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

        # -- Penalize angular velocity in XY plane (roll/pitch) --
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # -- Feet clearance reward --
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)

        ## === Use for flat terrain ===
        foot_height = self.foot_positions_w[:, :, 2]
        target_height = 0.08 * phases + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1.0 - self.desired_contact_states)
        # ===
        # === Use for rough terrain ===
        # # use foot height relative to base for clearance reward (better for uneven terrain)
        # base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
        # foot_z_rel = self.foot_positions_w[:, :, 2] - base_z
        # # target is "nominal stance height" + swing clearance bump
        # target_z_rel = self._nominal_foot_z_rel + (0.08 * phases + 0.02)
        # rew_foot_clearance = (target_z_rel - foot_z_rel).square() * (1.0 - self.desired_contact_states)
        # ===
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)

        # -- Contact tracking shaped by forces reward --
        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :], dim=-1)
        desired_contact = self.desired_contact_states
        rew_tracking_contacts_shaped_force = 0.

        for i in range(4):
            rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))        
        
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4 # avg over 4 feet

        # -- Torque Regularization --
        rew_torque = torch.sum(torch.square(self.torques), dim=1)

        # -- Foot2Contact Reward --
        # NOTE: Using the Z component of world-frame net_forces_w for feet
        foot_contact_forces_z = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2]
        
        # 1. Check which feet have contact force > 1.0 N (i.e., are in contact)
        contacts_binary = (foot_contact_forces_z > 1.0)
        
        # 2. Sum the contacts (count how many feet are in contact)
        num_contacts = contacts_binary.sum(1).float()
        
        # 3. Calculate the penalty: |num_contacts - 2| / 2
        #    - If num_contacts is 2, penalty is |2-2|/2 = 0 (max reward)
        #    - If num_contacts is 0 or 4, penalty is |0-2|/2 = 1 or |4-2|/2 = 1 (max penalty)
        rew_foot2contact = - torch.abs(num_contacts - 2) / 2.0

        # -- Feet air-time reward (encourages stepping when commanded) --
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]  # (E,4)
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]                  # (E,4)

        # reward a "proper" swing: only when the foot just touched down, and it was in the air for a while
        air_time_rew = torch.sum((last_air_time - 0.3) * first_contact, dim=1)

        # gate it: only pay this reward when the command wants motion
        cmd_speed = torch.norm(self._commands[:, :2], dim=1)
        air_time_rew = air_time_rew * (cmd_speed > 0.1)

        # Add to rewards dict
        # scale each reward component by coeffs from cfg
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "rew_torque": rew_torque * self.cfg.torque_reward_scale,
            "foot2contact": rew_foot2contact * self.cfg.foot2contact_reward_scale,
            "feet_air_time": air_time_rew * self.cfg.feet_air_time_reward_scale,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) # sum all reward components to get final reward per env
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _step_contact_targets(self):
        """Builds contact plan; builds a time varying gait/foot contact schedule using a Raibert heuristic."""
        frequencies = 3. # Hz
        phases = 0.5
        offsets = 0.
        bounds = 0.
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device) # foot in contact for 50% of the gait cycle
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0) # every env has track of its own gait phase [0, 1]

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        # Normalize to halves of cycle
        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # Sinusoidal clock inputs
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w),
                                                            cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: 
        # termination conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # terminate if undesired contact forces
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # terminate if robot is upside down
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        
        # terminate if base height is too low
        base_height = self.robot.data.root_pos_w[:, 2]
        # cstr_base_height_min = base_height < self.cfg.base_height_min

        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        # died = cstr_upsidedown
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None): 
        # called when envs done and need to be reset
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        self.last_actions[env_ids] = 0.
        # Reset raibert quantity
        self.gait_indices[env_ids] = 0
        # Draw new random friction parameters
        self.mu_v[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) * self.cfg.mu_v_lim).expand(-1, 12)
        self.F_s[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) * self.cfg.F_s_lim).expand(-1, 12)

        base_z = self.robot.data.root_pos_w[env_ids, 2].unsqueeze(1)
        feet_z = self.foot_positions_w[env_ids, :, 2]
        self._nominal_foot_z_rel[env_ids] = torch.mean(feet_z - base_z, dim=1, keepdim=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
