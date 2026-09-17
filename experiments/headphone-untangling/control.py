"""Execute substantial, feedback-monitored motions from a physical checkpoint.

Only stock actuator inputs move the robot and headphones. The canonical MJCF
and all collision geometry remain unchanged. Integration settings are recorded
with each run, and failed searches are kept distinct from accepted results.
"""

import hashlib
import json
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from environment import (
    ROOT,
    load,
    save,
    nodes,
    crossings,
    aloha_checks,
    geometry_contacts,
)
from strategy import solve_pose, endpoint_analysis, grasp_scan
from grip import smoothstep, grip_observation


class Rollout:
    def __init__(
        self,
        state,
        name,
        dt=0.0002,
        iterations=100,
        solver="CG",
        integrator=None,
        jacobian=None,
    ):
        self.model, self.data = load(state)
        self.model.opt.timestep = dt
        self.model.opt.iterations = iterations
        self.model.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + solver)
        if jacobian is not None:
            self.model.opt.jacobian = getattr(
                mujoco.mjtJacobian, "mjJAC_" + jacobian.upper()
            )
        if integrator is not None:
            self.model.opt.integrator = getattr(
                mujoco.mjtIntegrator, "mjINT_" + integrator.upper()
            )
        self.name, self.state = name, state
        self.start_time = self.data.time
        self.wall = time.monotonic()
        self.scratch = mujoco.MjData(self.model)
        self.arms = ["left", "right"]
        self.qadr = {
            s: np.array(
                [self.model.joint(s + "/" + j).qposadr[0] for j in aloha_checks.JOINTS]
            )
            for s in self.arms
        }
        self.dadr = {
            s: np.array(
                [self.model.joint(s + "/" + j).dofadr[0] for j in aloha_checks.JOINTS]
            )
            for s in self.arms
        }
        self.aids = {
            s: np.array(
                [self.model.actuator(s + "/" + j).id for j in aloha_checks.JOINTS]
            )
            for s in self.arms
        }
        self.targets = {s: self.data.qpos[self.qadr[s]].copy() for s in self.arms}
        self.integral = {s: np.zeros(6) for s in self.arms}
        self.fingers = {
            s: float(self.data.ctrl[self.model.actuator(s + "/gripper").id])
            for s in self.arms
        }
        previous = ROOT / "outputs" / (state + ".json")
        previous_report = json.loads(previous.read_text()) if previous.exists() else {}
        controller = previous_report.get("controller", {})
        for s in self.arms:
            if s in controller:
                self.targets[s] = np.asarray(controller[s]["target"])
                self.integral[s] = np.asarray(controller[s]["integral"])
            elif previous_report.get("phases"):
                # Preserve the last physical holding command when resuming an
                # older checkpoint that did not save its controller state.
                aids, dadr = self.aids[s], self.dadr[s]
                self.integral[s] = np.clip(
                    self.data.ctrl[aids]
                    - self.targets[s]
                    - self.data.qfrc_bias[dadr] / self.model.actuator_gainprm[aids, 0],
                    -0.025,
                    0.025,
                )
        self.held = previous_report.get("held", {})
        if not previous_report:
            for s in self.arms:
                for obj in ["plug", "left_earbud", "right_earbud"]:
                    if grip_observation(self.model, self.data, s, obj)[
                        "opposing_contacts"
                    ]:
                        self.held[s] = obj
        self.grip_reference = {
            s: grip_observation(self.model, self.data, s, o)["object_in_gripper"]
            for s, o in self.held.items()
        }
        self.grip_reference.update(previous_report.get("grip_reference", {}))
        # New cable grasps may monitor a material point at the pad center.
        # Older checkpoints retain their body-origin reference convention.
        self.grip_local_points = previous_report.get("grip_local_points", {})
        self.cable_geoms = np.array(
            ["_segment_" in self.model.geom(i).name for i in range(self.model.ngeom)]
        )
        self.geom_bodies = [
            self.model.body(self.model.geom_bodyid[i]).name
            for i in range(self.model.ngeom)
        ]
        self.allowed_grasp = {s: {o} for s, o in self.held.items()}
        for s, o in self.held.items():
            if "_cable_" in o:
                branch, index = o.rsplit("_cable_", 1)
                self.allowed_grasp[s] = {
                    f"{branch}_cable_{i:03}"
                    for i in range(max(0, int(index) - 2), int(index) + 3)
                }
        self.trajectory, self.observations, self.phases = [], [], []
        self.peak_penetration = 0.0
        self.error = None
        self.max_slip = 0.0
        self.initial = self.snapshot()
        save(self.data, name + "_start")
        self.record("start")
        self.robot_check = aloha_checks.robot_check(self.model)

    @property
    def now(self):
        return self.data.time - self.start_time

    def material_point_in_gripper(self, side, obj):
        body = self.data.body(obj)
        site = self.data.site(side + "/gripper")
        point = body.xpos + body.xmat.reshape(3, 3) @ np.asarray(
            self.grip_local_points.get(side, [0, 0, 0])
        )
        return site.xmat.reshape(3, 3).T @ (point - site.xpos)

    def track_cable_grip(self, side, branch, index):
        """Monitor a contacted cable midpoint; this creates no constraint."""
        obj = f"{branch}_cable_{index:03}"
        grip = grip_observation(self.model, self.data, side, obj)
        if not grip["opposing_contacts"]:
            self.error = f"No opposing contact on {obj}"
            return
        body = self.data.body(obj)
        geom = self.data.geom(f"{branch}_segment_{index:03}")
        self.grip_local_points[side] = (
            body.xmat.reshape(3, 3).T @ (geom.xpos - body.xpos)
        ).tolist()
        self.held[side] = obj
        self.grip_reference[side] = self.material_point_in_gripper(side, obj).tolist()
        self.allowed_grasp[side] = {
            f"{branch}_cable_{i:03}" for i in range(max(0, index - 2), index + 3)
        }

    def track_hardware_grip(self, side, obj):
        """Track the rigid material point between the actual opposing pads."""
        if not grip_observation(self.model, self.data, side, obj)["opposing_contacts"]:
            self.error = f"No opposing contact on {obj}"
            return
        points = []
        for c in self.data.contact:
            bodies = [self.geom_bodies[c.geom1], self.geom_bodies[c.geom2]]
            if obj in bodies and any(
                b in [side + "/left_finger_link", side + "/right_finger_link"]
                for b in bodies
            ):
                points.append(c.pos.copy())
        b = self.data.body(obj)
        self.grip_local_points[side] = (
            b.xmat.reshape(3, 3).T @ (np.mean(points, axis=0) - b.xpos)
        ).tolist()
        self.held[side] = obj
        self.allowed_grasp[side] = {obj}
        self.grip_reference[side] = self.material_point_in_gripper(side, obj).tolist()

    def snapshot(self):
        points = nodes(self.model, self.data)
        grips = {
            s: grip_observation(self.model, self.data, s, o)
            for s, o in self.held.items()
        }
        for s, o in self.held.items():
            grips[s]["tracked_material_point_in_gripper"] = (
                self.material_point_in_gripper(s, o).tolist()
            )
        return {
            "time": self.now,
            "projected_crossings": len(crossings(points)),
            "parts": {
                n: self.data.body(n).xpos.tolist()
                for n in ["plug", "left_earbud", "right_earbud", "splitter"]
            },
            "grips": grips,
        }

    def record(self, phase):
        self.trajectory.append(
            (
                self.now,
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.data.ctrl.copy(),
            )
        )

    def control(self, desired, velocities):
        m, d = self.model, self.data
        for s in self.arms:
            qadr, dadr, aids = self.qadr[s], self.dadr[s], self.aids[s]
            error = desired[s] - d.qpos[qadr]
            # A bounded outer integral removes stiction offsets. Compensation
            # is expressed entirely as commands to the original position servos.
            self.integral[s] = np.clip(
                self.integral[s] + 0.7 * error * m.opt.timestep, -0.025, 0.025
            )
            feedforward = (
                d.qfrc_bias[dadr]
                + m.dof_damping[dadr] * velocities[s]
                + m.dof_frictionloss[dadr] * np.tanh(velocities[s] / 0.01)
            )
            ctrl = (
                desired[s]
                + self.integral[s]
                + feedforward / m.actuator_gainprm[aids, 0]
            )
            d.ctrl[aids] = np.clip(
                ctrl, m.actuator_ctrlrange[aids, 0], m.actuator_ctrlrange[aids, 1]
            )
            a = m.actuator(s + "/gripper")
            d.ctrl[a.id] = np.clip(self.fingers[s], *a.ctrlrange)

    def monitor(self):
        for c in self.data.contact:
            if self.cable_geoms[c.geom1] and self.cable_geoms[c.geom2]:
                self.peak_penetration = max(self.peak_penetration, -float(c.dist))
            if c.dist < -0.0005:
                bodies = [self.geom_bodies[c.geom1], self.geom_bodies[c.geom2]]
                for side in self.arms:
                    if not any(b.startswith(side + "/") for b in bodies):
                        continue
                    allowed = bool(
                        self.allowed_grasp.get(side, set()).intersection(bodies)
                    ) and any(
                        b in [side + "/left_finger_link", side + "/right_finger_link"]
                        for b in bodies
                    )
                    if not allowed:
                        self.error = f"Unplanned {side} collision: {bodies}"
        if any(w.number for w in self.data.warning):
            self.error = "MuJoCo instability warning"
        if self.peak_penetration > 0.0002:
            self.error = "Cable self penetration exceeded 0.2 mm"
        for s, o in self.held.items():
            grip = grip_observation(self.model, self.data, s, o)
            slip = float(
                np.linalg.norm(
                    self.material_point_in_gripper(s, o) - self.grip_reference[s]
                )
            )
            self.max_slip = max(self.max_slip, slip)
            if slip > 0.004:
                self.error = (
                    f"{s} material slipped {slip * 1000:.1f} mm through the {o} grip"
                    if grip["opposing_contacts"]
                    else f"{s} lost the {o} grip (relative displacement {slip * 1000:.1f} mm)"
                )
        return self.error is None

    def move(
        self,
        label,
        endpoints,
        duration,
        settle=0.15,
        rotations=None,
        finger_targets=None,
        joint_targets=None,
    ):
        """Smooth simultaneous Cartesian motion, precomputed IK, live contacts."""
        if self.error:
            return
        rotations = rotations or {}
        m, d = self.model, self.data
        points0 = {s: d.site(s + "/gripper").xpos.copy() for s in endpoints}
        rots0 = {s: d.site(s + "/gripper").xmat.reshape(3, 3).copy() for s in endpoints}
        rotvec = {
            s: Rotation.from_matrix(
                np.array(rotations.get(s, rots0[s])) @ rots0[s].T
            ).as_rotvec()
            for s in endpoints
        }
        fingers0 = self.fingers.copy()
        self.scratch.qpos[:] = d.qpos
        grid = np.linspace(0, duration, max(3, int(duration / 0.015) + 1))
        paths = {s: [] for s in self.arms}
        seeds = {s: self.targets[s].copy() for s in self.arms}
        joints0 = {s: d.qpos[self.qadr[s]].copy() for s in (joint_targets or {})}
        for t in grid:
            for s in self.arms:
                if s in (joint_targets or {}):
                    seeds[s] = joints0[s] + (
                        np.asarray(joint_targets[s]) - joints0[s]
                    ) * smoothstep(t / duration)
                    self.scratch.qpos[self.qadr[s]] = seeds[s]
                elif s in endpoints:
                    p = points0[s] + (np.array(endpoints[s]) - points0[s]) * smoothstep(
                        t / duration
                    )
                    rotation = (
                        Rotation.from_rotvec(
                            smoothstep(t / duration) * rotvec[s]
                        ).as_matrix()
                        @ rots0[s]
                    )
                    sol = solve_pose(
                        m, self.scratch, s, p, rotation, seed=seeds[s], forward=False
                    )
                    if (
                        sol["position_error_m"] > 0.001
                        or sol["rotation_error_rad"] > 0.01
                    ):
                        self.error = f"Unreachable {label}: {s} IK error {sol['position_error_m']:.4f} m"
                        return
                    seeds[s] = np.array(sol["joints"])
                paths[s].append(seeds[s].copy())
            moving = set(endpoints) | set(joint_targets or {})
            if moving:
                # Free-space approaches only. This scratch configuration is
                # discarded; the actual trajectory runs through actuators.
                geometry_contacts(m, self.scratch)
                for c in self.scratch.contact:
                    if c.dist >= -0.0002:
                        continue
                    bodies = [self.geom_bodies[c.geom1], self.geom_bodies[c.geom2]]
                    planned = any(
                        bool(self.allowed_grasp.get(s, set()).intersection(bodies))
                        and any(
                            b in [s + "/left_finger_link", s + "/right_finger_link"]
                            for b in bodies
                        )
                        for s in moving
                    )
                    rigid_obstacle = "world" in bodies or all(
                        b.startswith(("left/", "right/")) for b in bodies
                    )
                    if (
                        rigid_obstacle
                        and not planned
                        and any(b.startswith(s + "/") for s in moving for b in bodies)
                    ):
                        self.error = f"Blocked free-space approach {label}: {bodies}"
                        return
        paths = {s: np.array(p) for s, p in paths.items()}
        speeds = {s: np.gradient(p, grid, axis=0) for s, p in paths.items()}
        if any(np.max(np.abs(v)) > 4.0 for v in speeds.values()):
            self.error = f"Discontinuous or too-fast joint path in {label}; replan the wrist posture"
            return
        phase = {
            "name": label,
            "start": self.now,
            "duration": duration,
            "settle": settle,
            "endpoints": {s: np.array(p).tolist() for s, p in endpoints.items()},
            "integration": {
                "timestep": m.opt.timestep,
                "solver": mujoco.mjtSolver(m.opt.solver).name,
                "iterations": m.opt.iterations,
                "integrator": mujoco.mjtIntegrator(m.opt.integrator).name,
                "jacobian": mujoco.mjtJacobian(m.opt.jacobian).name,
            },
        }
        self.phases.append(phase)
        print("Executing", json.dumps(phase), flush=True)
        count = round((duration + settle) / m.opt.timestep)
        record_every = max(1, round(0.025 / m.opt.timestep))
        observe_every = max(1, round(0.25 / m.opt.timestep))
        for step in range(count):
            t = step * m.opt.timestep
            for s, f in (finger_targets or {}).items():
                self.fingers[s] = fingers0[s] + (f - fingers0[s]) * smoothstep(
                    t / duration
                )
            desired = {
                s: np.array([np.interp(t, grid, paths[s][:, j]) for j in range(6)])
                for s in self.arms
            }
            velocity = {
                s: np.array([np.interp(t, grid, speeds[s][:, j]) for j in range(6)])
                if t < duration
                else np.zeros(6)
                for s in self.arms
            }
            self.control(desired, velocity)
            mujoco.mj_step(m, d)
            if not self.monitor():
                break
            if step % record_every == 0:
                self.record(label)
            if step % observe_every == 0:
                item = self.snapshot()
                item["phase"] = label
                self.observations.append(item)
                print(
                    json.dumps(
                        {
                            "t": round(self.now, 3),
                            "phase": label,
                            "crossings": item["projected_crossings"],
                            "plug": np.round(item["parts"]["plug"], 4).tolist(),
                            "slip_mm": round(self.max_slip * 1000, 3),
                            "penetration_mm": round(self.peak_penetration * 1000, 3),
                        }
                    ),
                    flush=True,
                )
        self.targets = {s: paths[s][-1].copy() for s in self.arms}
        mujoco.mj_forward(m, d)
        self.record(label)
        phase["finish"] = self.now
        phase["result"] = self.snapshot()
        save(d, self.name + "_" + str(len(self.phases)))
        self.write(phase_checkpoint=True)

    def acquire(self, side, object_name):
        """Approach, descend, close and test the second handle without staging."""
        candidates = grasp_scan(
            self.model,
            self.data,
            [(object_name, side)],
            grasp_mode="stem" if "earbud" in object_name else "housing",
        )[object_name]
        if not candidates:
            self.error = f"No collision-free opposing grasp candidate for {object_name}"
            return
        candidate = candidates[0]
        self.allowed_grasp[side] = {object_name}
        point = np.array(candidate["site_point"])
        self.move(
            "approach " + object_name,
            {side: point + [0, 0, 0.12]},
            1.4,
            0.1,
            rotations={side: candidate["rotation"]},
            finger_targets={side: candidate["finger_position"] + 0.003},
        )
        self.move("descend to " + object_name, {side: point}, 1.0, 0.15)
        self.move(
            "close on " + object_name,
            {},
            0.2,
            0.25,
            finger_targets={side: candidate["finger_position"] - 0.0003},
        )
        if self.error:
            return
        grip = grip_observation(self.model, self.data, side, object_name)
        if not grip["opposing_contacts"]:
            self.error = f"No opposing contact after closing on {object_name}"
            return
        self.held[side] = object_name
        self.grip_local_points.pop(side, None)
        self.grip_reference[side] = grip["object_in_gripper"]
        self.move(
            "lift " + object_name,
            {side: self.data.site(side + "/gripper").xpos + [0, 0, 0.05]},
            0.8,
            0.1,
        )

    def follow_and_pinch(self, side, branch, index, candidate, timeout=1.1):
        """Track a settling cable inside the pads while physically closing them.

        Only visual/material position feedback and stock actuators are used.
        The target is never welded or moved directly. This avoids relying on
        the cable staying still between an approach and a fixed-pose closure.
        """
        from strategy import fingertip_center

        if self.error:
            return
        m, d = self.model, self.data
        start = self.now
        obj = f"{branch}_cable_{index:03}"
        self.allowed_grasp[side] = {
            f"{branch}_cable_{i:03}" for i in range(max(0, index - 2), index + 3)
        }
        rotation = np.asarray(candidate["rotation"])
        command_rotation = d.site(side + "/gripper").xmat.reshape(3, 3).copy()
        command_point = d.site(side + "/gripper").xpos.copy()
        pad = fingertip_center(m, d, side)
        pad[1] = 0.0
        opened = candidate["finger_position"] + 0.006
        closed = candidate["finger_position"] - 0.0013
        finger0 = self.fingers[side]
        close_at = None
        opposed_since = None
        velocities = {s: np.zeros(6) for s in self.arms}
        refresh = max(1, round(0.02 / m.opt.timestep))
        phase = {
            "name": "track the settling cable and pinch it inside the pads",
            "start": start,
            "duration": timeout,
            "settle": 0,
            "endpoints": {},
            "feedback_grasp": True,
            "integration": {
                "timestep": m.opt.timestep,
                "solver": mujoco.mjtSolver(m.opt.solver).name,
                "iterations": m.opt.iterations,
                "integrator": mujoco.mjtIntegrator(m.opt.integrator).name,
                "jacobian": mujoco.mjtJacobian(m.opt.jacobian).name,
            },
        }
        self.phases.append(phase)
        print("Executing", json.dumps(phase), flush=True)
        success = False
        for step in range(round(timeout / m.opt.timestep)):
            t = self.now - start
            point = d.geom(f"{branch}_segment_{index:03}").xpos - rotation @ pad
            delta = point - d.site(side + "/gripper").xpos
            if step % refresh == 0:
                interval = refresh * m.opt.timestep
                change = Rotation.from_matrix(rotation @ command_rotation.T).as_rotvec()
                command_rotation = (
                    Rotation.from_rotvec(
                        change * min(1.0, interval / max(np.linalg.norm(change), 1e-12))
                    ).as_matrix()
                    @ command_rotation
                )
                desired_point = (
                    d.geom(f"{branch}_segment_{index:03}").xpos
                    - command_rotation @ pad
                    + 0.4 * delta
                )
                translation = desired_point - command_point
                command_point += translation * min(
                    1.0, 0.03 * interval / max(np.linalg.norm(translation), 1e-12)
                )
                self.scratch.qpos[:] = d.qpos
                sol = solve_pose(
                    m,
                    self.scratch,
                    side,
                    command_point,
                    command_rotation,
                    seed=self.targets[side],
                    forward=False,
                )
                if sol["position_error_m"] > 0.001 or sol["rotation_error_rad"] > 0.01:
                    self.error = "Unreachable cable tracking pose"
                    break
                proposed = np.asarray(sol["joints"])
                joint_change = proposed - self.targets[side]
                if np.max(np.abs(joint_change)) > 1.0:
                    self.error = "Discontinuous cable tracking pose"
                    break
                proposed = self.targets[side] + joint_change * min(
                    1.0, 2.0 * interval / max(np.max(np.abs(joint_change)), 1e-12)
                )
                velocity = (proposed - self.targets[side]) / interval
                self.scratch.qpos[self.qadr[side]] = proposed
                geometry_contacts(m, self.scratch)
                blocked = []
                for c in self.scratch.contact:
                    bodies = [self.geom_bodies[c.geom1], self.geom_bodies[c.geom2]]
                    rigid = "world" in bodies or all(
                        b.startswith(("left/", "right/")) for b in bodies
                    )
                    if (
                        c.dist < -0.0002
                        and rigid
                        and any(b.startswith(side + "/") for b in bodies)
                    ):
                        blocked.append(bodies)
                if blocked:
                    self.error = "Blocked cable tracking pose: " + str(blocked[0])
                    break
                self.targets[side] = proposed
                velocities[side] = velocity
            if close_at is None and t >= 0.10 and np.linalg.norm(delta) < 0.001:
                close_at = t
            if close_at is None:
                self.fingers[side] = finger0 + (opened - finger0) * smoothstep(t / 0.10)
            else:
                self.fingers[side] = opened + (closed - opened) * smoothstep(
                    (t - close_at) / 0.20
                )
            self.control(self.targets, velocities)
            mujoco.mj_step(m, d)
            if not self.monitor():
                break
            if step % max(1, round(0.025 / m.opt.timestep)) == 0:
                self.record(phase["name"])
            if step % max(1, round(0.25 / m.opt.timestep)) == 0:
                observation = self.snapshot()
                observation["phase"] = phase["name"]
                self.observations.append(observation)
                print(
                    json.dumps(
                        {
                            "t": self.now,
                            "tracking_error_m": float(np.linalg.norm(delta)),
                            "closure_started": close_at is not None,
                        }
                    ),
                    flush=True,
                )
            grip = grip_observation(m, d, side, obj)
            local_point = d.site(side + "/gripper").xmat.reshape(3, 3).T @ (
                d.geom(f"{branch}_segment_{index:03}").xpos
                - d.site(side + "/gripper").xpos
            )
            seated = local_point[0] < pad[0] + 0.0008
            if (
                close_at is not None
                and t - close_at >= 0.20
                and grip["opposing_contacts"]
                and seated
            ):
                if opposed_since is None:
                    opposed_since = t
                if t - opposed_since >= 0.08:
                    success = True
                    break
            else:
                opposed_since = None
        mujoco.mj_forward(m, d)
        if not self.error:
            if not success:
                self.error = "No sustained opposing contact during tracked closure"
            else:
                self.track_cable_grip(side, branch, index)
        self.record(phase["name"])
        phase["finish"] = self.now
        phase["result"] = self.snapshot()
        save(d, self.name + "_" + str(len(self.phases)))
        self.write(phase_checkpoint=True)

    def release(self, side, position=None):
        if self.error:
            return
        obj = self.held[side]
        body = self.data.body(obj)
        target = np.array(position) if position is not None else body.xpos.copy()
        if position is None:
            target[2] = 0.019
        point = self.data.site(side + "/gripper").xpos + target - body.xpos
        self.move("place " + obj, {side: point}, 0.9, 0.15)
        if self.error:
            return
        del self.held[side]
        del self.grip_reference[side]
        self.grip_local_points.pop(side, None)
        self.move("release " + obj, {}, 0.25, 0.15, finger_targets={side: 0.025})
        self.move(
            "withdraw from " + obj,
            {side: self.data.site(side + "/gripper").xpos + [0, 0, 0.08]},
            0.6,
            0.1,
        )

    def acquire_cord(self, side, branch, index, lift=0.065):
        from cord_scan import scan

        choices = scan(self.model, self.data, branch, index, side)
        if not choices:
            self.error = f"No accessible cord grasp for {branch} {index}"
            return
        current = self.data.qpos[self.qadr[side]]
        candidate = min(
            choices, key=lambda c: np.linalg.norm(np.array(c["joints"]) - current)
        )
        obj = candidate["target_body"]
        self.allowed_grasp[side] = {
            f"{branch}_cable_{i:03}" for i in range(max(0, index - 2), index + 3)
        }
        point = np.array(candidate["site_point"])
        self.move(
            "approach the blocking loop",
            {side: point + [0, 0, 0.08]},
            1.0,
            0.1,
            rotations={side: candidate["rotation"]},
            finger_targets={side: candidate["finger_position"] + 0.0018},
        )
        self.move("descend onto the loop", {side: point}, 0.7, 0.15)
        self.move(
            "pinch the loop",
            {},
            0.2,
            0.25,
            finger_targets={side: candidate["finger_position"] - 0.0003},
        )
        if self.error:
            return
        grip = grip_observation(self.model, self.data, side, obj)
        if not grip["opposing_contacts"]:
            self.error = "No opposing finger load on the loop"
            return
        self.track_cable_grip(side, branch, index)
        self.move(
            "open the loop vertically",
            {side: self.data.site(side + "/gripper").xpos + [0, 0, lift]},
            0.9,
            0.15,
        )

    def write(self, phase_checkpoint=False):
        save(self.data, self.name)
        np.savez_compressed(
            ROOT / "outputs" / (self.name + "_trajectory.npz"),
            **{
                k: np.array([x[i] for x in self.trajectory])
                for i, k in enumerate(["time", "qpos", "qvel", "ctrl"])
            },
        )
        report = {
            "start_state": self.state,
            "scene_sha256": hashlib.sha256(
                (ROOT / "scene.xml").read_bytes()
            ).hexdigest(),
            "simulation_s": self.now,
            "wall_s": time.monotonic() - self.wall,
            "error": self.error,
            "integration": {
                "timestep": self.model.opt.timestep,
                "solver": mujoco.mjtSolver(self.model.opt.solver).name,
                "iterations": self.model.opt.iterations,
                "integrator": mujoco.mjtIntegrator(self.model.opt.integrator).name,
                "jacobian": mujoco.mjtJacobian(self.model.opt.jacobian).name,
            },
            "robot_unmodified": True,
            "uses_ground_truth": True,
            "gripper_object_weld": False,
            "staged_initial_grasp": self.state.startswith("grasp_"),
            "untangled": False,
            "held": self.held,
            "grip_reference": self.grip_reference,
            "grip_local_points": self.grip_local_points,
            "controller": {
                s: {
                    "target": self.targets[s].tolist(),
                    "integral": self.integral[s].tolist(),
                }
                for s in self.arms
            },
            "peak_cable_penetration_m": self.peak_penetration,
            "max_grip_displacement_m": self.max_slip,
            "initial": self.initial,
            "final": self.snapshot(),
            "phases": self.phases,
            "observations": self.observations,
            "endpoint_analysis": endpoint_analysis(self.model, self.data),
        }
        (ROOT / "outputs" / (self.name + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if phase_checkpoint and self.phases and "finish" in self.phases[-1]:
            # Pair each physical phase checkpoint with its exact controller
            # state, so a later rejected motion need not repeat good actions.
            (
                ROOT / "outputs" / (self.name + "_" + str(len(self.phases)) + ".json")
            ).write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in [
                        "simulation_s",
                        "wall_s",
                        "error",
                        "peak_cable_penetration_m",
                        "max_grip_displacement_m",
                    ]
                }
            ),
            flush=True,
        )
