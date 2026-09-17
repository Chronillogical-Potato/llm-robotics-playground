"""Trusted firmware. Pilots set body velocity and Cartesian hand targets only."""

from dataclasses import dataclass
import math
import numpy as np
import mujoco


@dataclass
class Drive:
    forward: float = 0
    turn: float = 0
    height: float = 0.44


class Firmware:
    def __init__(self, m, d):
        self.m = m
        self.d = d
        self.drive = Drive()
        self.phase = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.height = 0.44
        self.legs = []
        for s in [-1, 1]:
            for k, x in enumerate([-0.325, 0, 0.325]):
                names = [f"leg_{s}_{k}_{j}" for j in ["yaw", "hip", "knee"]]
                self.legs.append(
                    (
                        s,
                        k,
                        x,
                        s * math.radians([112, 90, 68][k]),
                        [m.actuator(n).id for n in names],
                    )
                )
        self.scratch = mujoco.MjData(m)
        self.hands = {}
        for side in ["left", "right"]:
            js = [m.joint(f"{side}_arm_joint_{i}") for i in range(1, 8)]
            self.hands[side] = {
                "q": [j.qposadr[0] for j in js],
                "v": [j.dofadr[0] for j in js],
                "a": [m.actuator(f"{side}_arm_joint_{i}").id for i in range(1, 8)],
                "site": m.site(side + "_grip_pinch").id,
                "goal": None,
                "grip": 0.0,
                "frame": "world",
            }
        self.tick = 0
        for h in self.hands.values():
            h["velocity"] = np.zeros(7)
            h["qgoal"] = d.ctrl[h["a"]].copy()
            h["trajectory"] = None

    def command_drive(self, forward=0, turn=0, height=0.44):
        if not (
            -0.35 <= forward <= 0.35 and -0.4 <= turn <= 0.4 and 0.36 <= height <= 0.51
        ):
            raise ValueError("Drive outside budget")
        self.drive = Drive(forward, turn, height)

    def command_hand(self, side, position, rotation, grip=None, frame="world"):
        if side not in self.hands or frame not in ["world", "body"]:
            raise ValueError("Invalid hand/frame")
        p = np.array(position, dtype=float)
        R = np.array(rotation, dtype=float).reshape(3, 3)
        if (
            p.shape != (3,)
            or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(R))
            or not np.allclose(R.T @ R, np.eye(3), atol=0.001)
            or np.linalg.det(R) < 0.99
        ):
            raise ValueError("Invalid hand pose")
        h = self.hands[side]
        changed = (
            h["goal"] is None
            or h["frame"] != frame
            or not np.allclose(h["goal"][0], p, atol=1e-7)
            or not np.allclose(h["goal"][1], R, atol=1e-7)
        )
        if changed:
            p0, R0 = self.hand_pose(side)
            if frame == "body":
                b = self.d.body("robot")
                B = b.xmat.reshape(3, 3)
                p0 = B.T @ (p0 - b.xpos)
                R0 = B.T @ R0
            angle = math.acos(np.clip((np.trace(R @ R0.T) - 1) / 2, -1, 1))
            duration = max(1.2, float(np.linalg.norm(p - p0)) / 0.065, angle / 0.4)
            h["trajectory"] = (self.d.time, duration, p0, R0)
        h["goal"] = (p, R)
        h["frame"] = frame
        if grip is not None:
            if not 0 <= grip <= 1:
                raise ValueError("Grip outside [0,1]")
            h["grip"] = grip

    def hand_pose(self, side):
        h = self.hands[side]
        return self.d.site_xpos[h["site"]].copy(), self.d.site_xmat[h["site"]].reshape(
            3, 3
        ).copy()

    def solve(self, h, pos, rot):
        s = self.scratch
        s.qpos[:] = self.d.qpos
        s.qpos[h["q"]] = self.d.ctrl[h["a"]]
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros_like(jp)
        for _ in range(100):
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            current = s.site_xmat[h["site"]].reshape(3, 3)
            er = 0.5 * sum(
                (np.cross(current[:, i], rot[:, i]) for i in range(3)), np.zeros(3)
            )
            ep = pos - s.site_xpos[h["site"]]
            err = np.r_[ep, 0.35 * er]
            if np.linalg.norm(ep) < 0.001 and np.linalg.norm(er) < 0.008:
                break
            mujoco.mj_jacSite(self.m, s, jp, jr, h["site"])
            J = np.vstack([jp[:, h["v"]], 0.35 * jr[:, h["v"]]])
            delta = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * 0.0001, err)
            s.qpos[h["q"]] += np.clip(delta, -0.12, 0.12)
            s.qpos[h["q"]] = np.clip(
                s.qpos[h["q"]],
                self.m.actuator_ctrlrange[h["a"], 0],
                self.m.actuator_ctrlrange[h["a"], 1],
            )
        return s.qpos[h["q"]].copy()

    def step(self):
        dt = self.m.opt.timestep
        d = self.d
        accel = 0.12 if max(abs(self.drive.forward), abs(self.speed)) > 0.13 else 0.04
        self.speed += np.clip(self.drive.forward - self.speed, -accel * dt, accel * dt)
        self.turn += np.clip(self.drive.turn - self.turn, -0.12 * dt, 0.12 * dt)
        self.height += np.clip(self.drive.height - self.height, -0.03 * dt, 0.03 * dt)
        moving = abs(self.speed) + abs(self.turn) > 0.0001
        # Fast cadence during cruising, slower support for tight turns and docking.
        cruise = np.clip((abs(self.drive.forward) - 0.055) / 0.035, 0, 1) * np.clip(
            1 - abs(self.drive.turn) / 0.18, 0, 1
        )
        fast = np.clip((abs(self.speed) - 0.13) / 0.12, 0, 1)
        target_period = (1.5 - 1.0 * cruise) * (1 - fast) + 0.85 * fast
        self.gait_period = getattr(self, "gait_period", 1.5) + np.clip(
            target_period - getattr(self, "gait_period", 1.5), -0.5 * dt, 0.5 * dt
        )
        period = self.gait_period
        duty = 0.68 - 0.08 * fast
        self.phase = (self.phase + dt / period) % 1
        self.gait_blend = getattr(self, "gait_blend", 0.0) + np.clip(
            float(moving) - getattr(self, "gait_blend", 0.0), -dt * 2, dt * 2
        )
        offsets = [0, 0.5, 0, 0.5, 0, 0.5]
        for idx, (s, k, x, theta, a) in enumerate(self.legs):
            phase = (self.phase + offsets[idx]) % 1
            fx = 0.41875 * math.cos(theta)
            fy = 0.41875 * math.sin(theta)
            vx = self.speed - self.turn * (s * 0.33 + fy)
            vy = self.turn * (x + fx)
            if phase < duty:
                u = 0.5 - phase / duty
                lift = 0
            else:
                q = (phase - duty) / (1 - duty)
                ease = q**3 * (10 - 15 * q + 6 * q * q)
                v = -(1 - duty) / duty
                u = -0.5 + v * q + (1 - v) * ease
                lift = 0.045 * math.sin(math.pi * q) ** 2 * self.gait_blend
            fx += vx * period * duty * u
            fy += vy * period * duty * u
            yaw = math.atan2(fy, fx) - theta
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            rr = math.hypot(fx, fy) - 0.03125
            zz = -self.height + 0.0225 + 0.025 + lift
            L1 = math.hypot(0.2625, 0.03125)
            L2 = math.hypot(0.125, 0.42125)
            bend = -math.acos(
                np.clip((rr * rr + zz * zz - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1)
            )
            upper = math.atan2(zz, rr) - math.atan2(
                L2 * math.sin(bend), L1 + L2 * math.cos(bend)
            )
            hip = math.atan2(0.03125, 0.2625) - upper
            knee = math.atan2(-0.42125, 0.125) - math.atan2(0.03125, 0.2625) - bend
            d.ctrl[a] = [yaw, hip, knee]
        for side, h in self.hands.items():
            if h["goal"] is not None and self.tick % 5 == 0:
                pos, rot = h["goal"]
                start, duration, p0, R0 = h["trajectory"]
                u = np.clip((d.time - start) / duration, 0, 1)
                blend = u * u * u * (10 + u * (-15 + 6 * u))
                pos = p0 + (pos - p0) * blend
                q0 = np.zeros(4)
                q1 = np.zeros(4)
                mujoco.mju_mat2Quat(q0, R0.ravel())
                mujoco.mju_mat2Quat(q1, rot.ravel())
                dot = np.dot(q0, q1)
                if dot < 0:
                    q1 = -q1
                    dot = -dot
                if dot > 0.9995:
                    q = q0 + (q1 - q0) * blend
                    q /= np.linalg.norm(q)
                else:
                    angle = math.acos(np.clip(dot, -1, 1))
                    q = (
                        math.sin((1 - blend) * angle) * q0
                        + math.sin(blend * angle) * q1
                    ) / math.sin(angle)
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, q)
                rot = rot.reshape(3, 3)
                if h["frame"] == "body":
                    body = d.body("robot")
                    B = body.xmat.reshape(3, 3)
                    pos = body.xpos + B @ pos
                    rot = B @ rot
                h["qgoal"] = self.solve(h, pos, rot)
            desired_velocity = np.clip((h["qgoal"] - d.ctrl[h["a"]]) * 12, -0.65, 0.65)
            h["velocity"] += np.clip(
                desired_velocity - h["velocity"], -1.5 * dt, 1.5 * dt
            )
            d.ctrl[h["a"]] += h["velocity"] * dt
            aid = self.m.actuator(side + "_grip_fingers_actuator").id
            if h["grip"] < h.get("last_grip_command", 0.0):
                # Remove saturated target overtravel on opening, preserving current force.
                gain = self.m.actuator_gainprm[aid, 0]
                b = self.m.actuator_biasprm[aid]
                bias = (
                    b[0]
                    + b[1] * d.actuator_length[aid]
                    + b[2] * d.actuator_velocity[aid]
                )
                equivalent = (d.actuator_force[aid] - bias) / gain
                d.ctrl[aid] = min(
                    d.ctrl[aid], np.clip(equivalent, *self.m.actuator_ctrlrange[aid])
                )
            h["last_grip_command"] = h["grip"]
            d.ctrl[aid] += np.clip(h["grip"] * 255 - d.ctrl[aid], -160 * dt, 160 * dt)
        self.tick += 1
