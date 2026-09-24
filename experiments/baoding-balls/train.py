"""Train the fixed Baoding PPO task in Isaac Lab/PhysX (Linux + NVIDIA GPU)."""

import argparse
import copy
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
GOAL_SPEED = 1.2
ACTION_SCALE = .025
EPISODE_SECONDS = 12.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assets", type=Path, required=True,
                    help="Pinned Sharpa asset checkout (contains wave_01/right_sharpa_wave/*.usda)")
parser.add_argument("--out", type=Path, default=HERE / "outputs" / "train")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=200, help="PPO updates; 32 steps/env per update")
parser.add_argument("--seed", type=int, default=170921)
parser.add_argument("--resume", type=Path, help="Optional checkpoint; baseline.pt is the selected warm start")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1:
    parser.error("--num-envs and --iterations must be positive")
asset_file = args.assets.resolve() / "wave_01/right_sharpa_wave/right_sharpa_wave.usda"
if not asset_file.is_file():
    parser.error(f"Sharpa USD missing: {asset_file}")
if args.resume and not args.resume.is_file():
    parser.error(f"Checkpoint missing: {args.resume}")
if args.out.exists() and any(args.out.iterdir()):
    parser.error(f"Output folder is not empty: {args.out}; choose a fresh --out")

app = AppLauncher(args).app

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import isaaclab.sim as sim
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module

from bounds_ppo import BoundsPPO
from training_core import (advance_progress, geometry_and_risk, local_ball_coordinates,
                           middle_posture_risk, middle_side_risk, motion_costs)


HAND_POSITION = (0., 0., .25)
HAND_QUATERNION = (0., math.sqrt(.5), 0., math.sqrt(.5))
BALL_POSITIONS = ((.085, -.02, .302), (.085, .02, .302))
START_POSE = {"right_thumb_CMC_FE": .75, "right_thumb_CMC_AA": .05,
              "right_thumb_MCP_FE": .35, "right_thumb_MCP_AA": 0.,
              "right_thumb_IP": .55, "right_pinky_CMC": .08}
for finger in ("index", "middle", "ring", "pinky"):
    START_POSE.update({f"right_{finger}_MCP_FE": .45, f"right_{finger}_MCP_AA": 0.,
                       f"right_{finger}_PIP": .8, f"right_{finger}_DIP": .45})


@configclass
class BaodingCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = EPISODE_SECONDS
    action_space = 22
    observation_space = 107
    state_space = 0
    sim = sim.SimulationCfg(
        dt=1 / 240, render_interval=4,
        physics_material=sim.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.),
        physx=sim.PhysxCfg(bounce_threshold_velocity=.2, gpu_max_rigid_contact_count=2**23),
    )
    scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=.65, replicate_physics=True)
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_file), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=True,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_POSITION, rot=HAND_QUATERNION, joint_pos=START_POSE),
        actuators={"hand": ImplicitActuatorCfg(joint_names_expr=[".*"],
            stiffness={".*MCP_FE": 4.76, ".*MCP_AA": 6.62, ".*PIP": .9, ".*DIP": .9,
                       "right_thumb_IP": .9, "right_thumb_CMC_FE": 6.95,
                       "right_thumb_CMC_AA": 13.2, "right_pinky_CMC": 1.38},
            damping=None)},
    )


class BaodingEnv(DirectRLEnv):
    cfg: BaodingCfg

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.balls = []
        self.contacts = []
        for index in range(2):
            ball = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Ball{index}",
                spawn=sim.SphereCfg(
                    radius=.019,
                    rigid_props=sim.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.,
                        solver_position_iteration_count=16, solver_velocity_iteration_count=4),
                    mass_props=sim.MassPropertiesCfg(mass=.035),
                    collision_props=sim.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
                    physics_material=sim.RigidBodyMaterialCfg(
                        static_friction=1., dynamic_friction=1., restitution=0.),
                    visual_material=sim.PreviewSurfaceCfg(
                        diffuse_color=(.08, .45, .8) if index == 0 else (.94, .34, .07)),
                    activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_POSITIONS[index])))
            self.balls.append(ball)
            # Total ball force is a training proxy; it is not a hand-only contact audit.
            self.contacts.append(ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Ball{index}", update_period=0.,
                filter_prim_paths_expr=[], max_contact_data_count_per_prim=4)))
        sim.spawn_ground_plane("/World/ground", sim.GroundPlaneCfg(color=(.8, .8, .8)))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        for index, ball in enumerate(self.balls):
            self.scene.rigid_objects[f"ball{index}"] = ball
            self.scene.sensors[f"contact{index}"] = self.contacts[index]
        light = sim.DomeLightCfg(intensity=2200., color=(.9, .9, .9))
        light.func("/World/Light", light)

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        count, device = self.num_envs, self.device
        assert self.hand.num_joints == 22, self.hand.joint_names
        self.middle_flexion_ids = [self.hand.joint_names.index("right_middle_" + name)
                                   for name in ("MCP_FE", "PIP", "DIP")]
        self.middle_link_ids = [self.hand.body_names.index("right_middle_" + name)
                                for name in ("PP", "MP", "DP")]
        self.targets = self.hand.data.default_joint_pos.clone()
        self.actions = torch.zeros(count, 22, device=device)
        self.raw_actions = self.actions.clone()
        self.last_actions = self.actions.clone()
        self.lower = self.hand.data.soft_joint_pos_limits[:, :, 0]
        self.upper = self.hand.data.soft_joint_pos_limits[:, :, 1]
        self.center = torch.tensor([.085, 0., .302], device=device)
        self.phase = torch.zeros(count, device=device)
        self.progress = torch.zeros(count, 3, device=device)
        self.high = torch.zeros(count, device=device)
        self.old_angles = torch.zeros(count, 3, device=device)
        self.rewarded_progress = torch.zeros(count, device=device)
        self.held = torch.ones(count, dtype=torch.bool, device=device)
        self.goals = torch.zeros(count, 2, 3, device=device)
        self.settled = None
        self.calibrating = True
        self._update_state()
        self.old_angles[:] = self.angles

    def _update_state(self):
        self.position = torch.stack(
            [ball.data.root_pos_w - self.scene.env_origins for ball in self.balls], 1)
        self.velocity = torch.stack([ball.data.root_lin_vel_w for ball in self.balls], 1)
        radial = self.position[:, :, :2] - self.center[:2]
        pair = self.position[:, 1, :2] - self.position[:, 0, :2]
        self.angles = torch.cat((torch.atan2(radial[:, :, 1], radial[:, :, 0]),
                                 torch.atan2(pair[:, 1], pair[:, 0])[:, None]), 1)
        phase = self.phase[:, None] + torch.tensor([-math.pi / 2, math.pi / 2], device=self.device)
        self.goals[:, :, 0] = self.center[0] + .021 * torch.cos(phase)
        self.goals[:, :, 1] = self.center[1] + .021 * torch.sin(phase)
        self.goals[:, :, 2] = self.center[2]

    def _pre_physics_step(self, actions):
        self.last_actions[:] = self.actions
        self.raw_actions = actions.clone()
        self.actions = actions.clamp(-1, 1).clone()
        self.targets = (self.targets + ACTION_SCALE * self.actions).clamp(self.lower, self.upper)
        # The reference begins moving after one second; it never actuates either ball.
        self.phase += (self.episode_length_buf >= 60).float() * GOAL_SPEED * self.step_dt

    def _apply_action(self):
        self.hand.set_joint_position_target(self.targets)

    def _get_dones(self):
        self._update_state()
        radius = torch.linalg.vector_norm(self.position[:, :, :2] - self.center[:2], dim=-1)
        self.held = ((self.position[:, :, 2] > .258).all(1) &
                     (self.position[:, :, 2] < .37).all(1) & (radius < .072).all(1))
        self.sample_q = self.hand.data.joint_pos.clone()
        self.sample_joint_velocity = self.hand.data.joint_vel.clone()
        local = local_ball_coordinates(
            self.position + self.scene.env_origins[:, None, :],
            self.hand.data.body_link_pos_w[:, self.middle_link_ids],
            self.hand.data.body_link_quat_w[:, self.middle_link_ids])
        self.sample_middle_side = middle_side_risk(local)
        finite = torch.isfinite(self.position).all((1, 2)) & torch.isfinite(self.sample_q).all(1)
        return ~self.held | ~finite, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        support = torch.stack(
            [contact.data.net_forces_w.norm(dim=-1).sum(1) for contact in self.contacts], 1)
        geometry, supported, risk = geometry_and_risk(
            self.position, self.velocity, self.center, support)
        eligible = self.held & supported & geometry & (self.sample_middle_side <= .25)
        eligible &= self.episode_length_buf >= 60
        angular_delta = torch.atan2(torch.sin(self.angles - self.old_angles),
                                    torch.cos(self.angles - self.old_angles))
        self.progress, self.high, delta = advance_progress(
            self.old_angles, self.angles, self.progress, self.high, eligible,
            max_gain=GOAL_SPEED * self.step_dt)
        self.old_angles[:] = self.angles
        self.rewarded_progress += delta
        distance = torch.linalg.vector_norm(self.position - self.goals, dim=-1).mean(1)
        reward = 20. * delta + .3 * torch.exp(-distance / .015) * self.held
        reward -= 50. * (~self.held)
        reward += .2 * self.held - .5 * (~geometry) - .2 * (~supported) - .2 * risk
        costs = motion_costs(self.actions, self.last_actions,
                             self.hand.data.joint_vel, angular_delta, self.step_dt, GOAL_SPEED)
        radius = (self.position[:, :, :2] - self.center[:2]).norm(dim=-1)
        pair_xy = (self.position[:, 1, :2] - self.position[:, 0, :2]).norm(dim=-1)
        costs["angular_overspeed"] *= (radius > .008).all(1) & (pair_xy > .025)
        reward -= sum(costs.values())
        reward -= .2 * self.sample_middle_side * self.held
        if self.settled is not None:
            posture = middle_posture_risk(
                self.sample_q[:, self.middle_flexion_ids], self.settled["q"][self.middle_flexion_ids])
            reward -= .03 * posture * self.held
        self.extras["log"] = {
            "task/net_turns": self.progress.min(1).values.mean() / (2 * math.pi),
            "task/held_fraction": self.held.float().mean(),
            "task/target_error_m": distance.mean(),
            "task/geometry_fraction": geometry.float().mean(),
            "task/middle_side_risk": self.sample_middle_side.mean(),
            "control/action_clip_fraction": (self.raw_actions.abs() > 1).float().mean(),
        }
        return reward

    def _get_observations(self):
        self._update_state()
        observation = torch.cat((
            2 * (self.hand.data.joint_pos - self.lower) / (self.upper - self.lower) - 1,
            .1 * self.hand.data.joint_vel, self.targets, self.actions,
            (self.position - self.center).flatten(1) * 10,
            self.velocity.flatten(1),
            (self.goals - self.center).flatten(1) * 10,
            torch.remainder(self.phase, 2 * math.pi)[:, None] / math.pi - 1,
        ), 1)
        assert observation.shape[1] == 107, observation.shape
        return {"policy": observation}

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if ids is None:
            ids = self.hand._ALL_INDICES
        joints = self.hand.data.default_joint_pos[ids].clone()
        if self.settled is not None:
            joints[:] = self.settled["q"]
        if not self.calibrating:
            joints += .015 * torch.randn_like(joints)
        self.hand.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
        self.targets[ids] = joints
        self.hand.set_joint_position_target(joints, env_ids=ids)
        for index, ball in enumerate(self.balls):
            state = ball.data.default_root_state[ids].clone()
            if self.settled is not None:
                state[:, :7] = self.settled["balls"][index]
            state[:, :3] += self.scene.env_origins[ids]
            if not self.calibrating:
                state[:, :2] += .0005 * torch.randn_like(state[:, :2])
            ball.write_root_state_to_sim(state, env_ids=ids)
        self.phase[ids] = 0
        self.progress[ids] = 0
        self.high[ids] = 0
        self.rewarded_progress[ids] = 0
        self.actions[ids] = 0
        self.raw_actions[ids] = 0
        self.last_actions[ids] = 0
        self._update_state()
        self.old_angles[ids] = self.angles[ids]


def ppo_config():
    return dict(
        seed=args.seed, num_steps_per_env=32, save_interval=50,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        policy=dict(class_name="ActorCritic", init_noise_std=.35,
                    actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128],
                    activation="elu", actor_obs_normalization=True,
                    critic_obs_normalization=True),
        algorithm=dict(class_name="BoundsPPO", value_loss_coef=1.,
                       use_clipped_value_loss=True, clip_param=.2, entropy_coef=0.,
                       num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4,
                       schedule="fixed", gamma=.99, lam=.95, desired_kl=.01,
                       max_grad_norm=1., bounds_coef=.0001),
        logger="tensorboard",
    )


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = BaodingCfg()
    cfg.seed = args.seed
    cfg.viewer.eye = (.40, -.40, .64)
    cfg.viewer.lookat = (.08, 0., .28)
    env = BaodingEnv(cfg)
    try:
        env.reset()
        # Fixed static-grasp initialization. No policy reward is counted here.
        for _ in range(480):
            env.hand.set_joint_position_target(env.targets)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        env._update_state()
        initial = env.position.clone()
        env.center[:] = initial[0].mean(0)
        settled = {"q": env.hand.data.joint_pos[0].clone(), "center": env.center.clone(),
                   "balls": torch.stack([ball.data.root_state_w[0, :7].clone()
                                         for ball in env.balls])}
        settled["balls"][:, :3] -= env.scene.env_origins[0]
        if not ((initial[:, :, 2] > .258) & (initial[:, :, 2] < .37)).all():
            raise RuntimeError("Static grasp failed; refusing to train")
        support = torch.stack(
            [contact.data.net_forces_w.norm(dim=-1).sum(1) for contact in env.contacts], 1)
        if not (support > .01).all():
            raise RuntimeError("Static hand contact check failed; refusing to train")
        if args.resume:
            settled = torch.load(args.resume.parent / "settled.pt",
                                 map_location=env.device, weights_only=True)
            env.center[:] = settled["center"]
        env.settled = settled
        torch.save(settled, args.out / "settled.pt")
        (args.out / "configuration.json").write_text(json.dumps({
            "seed": args.seed, "num_envs": args.num_envs, "iterations": args.iterations,
            "episode_seconds": EPISODE_SECONDS, "goal_speed_rad_s": GOAL_SPEED,
            "action_scale": ACTION_SCALE, "assets": str(asset_file),
            "resume": str(args.resume) if args.resume else None,
        }, indent=2) + "\n")
        env.calibrating = False
        env.reset()
        config = ppo_config()
        (args.out / "ppo_config.json").write_text(json.dumps(config, indent=2) + "\n")
        runner_module.BoundsPPO = BoundsPPO
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = OnPolicyRunner(wrapper, copy.deepcopy(config), log_dir=str(args.out), device=env.device)
        if args.resume:
            runner.load(str(args.resume), load_optimizer=True)
        with torch.no_grad():
            runner.alg.policy.std.fill_(.35)
        runner.alg.policy.std.requires_grad_(False)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        runner.save(str(args.out / "final.pt"))
        print(f"Saved {args.out / 'final.pt'} at iteration {runner.current_learning_iteration}")
    finally:
        env.close()


try:
    main()
finally:
    app.close()
