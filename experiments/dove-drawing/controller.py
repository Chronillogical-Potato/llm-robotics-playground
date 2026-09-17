"""GPT-authored scripted controller. Task intelligence is outside fixed firmware."""

from pathlib import Path
import numpy as np
import mujoco
from firmware import Firmware

ROOT = Path(__file__).resolve().parent


class Controller:
    def __init__(self, firmware=None):
        self.f = firmware or Firmware()
        self.f.reset()
        self.m = self.f.model
        self.ik = mujoco.MjData(self.m)
        self.sid = self.m.site("grip_frame").id
        self.actions = []
        self.nominal = self.f.data.qpos[:9].copy()

    def pose(self):
        return self.f.data.site_xpos[self.sid].copy(), self.f.data.site_xmat[
            self.sid
        ].reshape(3, 3).copy()

    def solve(self, xyz, rot, seed=None):
        m, d = self.m, self.ik
        d.qpos[:] = self.f.data.qpos
        if seed is not None:
            d.qpos[:9] = seed
        for _ in range(100):
            mujoco.mj_forward(m, d)
            R = d.site_xmat[self.sid].reshape(3, 3)
            ep = xyz - d.site_xpos[self.sid]
            er = sum((np.cross(R[:, i], rot[:, i]) for i in range(3))) * 0.5
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            jp = np.zeros((3, m.nv))
            jr = jp.copy()
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack((jp[:, :9], jr[:, :9] * 0.3))
            err = np.r_[ep, er * 0.3]
            pin = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * 0.00001, np.eye(6))
            dq = pin @ err + (np.eye(9) - pin @ J) @ (
                0.005 * (self.nominal - d.qpos[:9])
            )
            d.qpos[:9] += np.clip(dq, -0.06, 0.06)
            d.qpos[:9] = np.clip(
                d.qpos[:9], m.jnt_range[:9, 0] + 0.002, m.jnt_range[:9, 1] - 0.002
            )
        mujoco.mj_forward(m, d)
        return d.qpos[:9].copy(), float(np.linalg.norm(xyz - d.site_xpos[self.sid]))

    def command(self, targets, duration):
        self.actions.append(
            {"time_s": self.f.data.time, "targets": targets, "duration_s": duration}
        )
        return self.f.command(targets, duration)

    def move(self, xyz, seconds=1, rot=None):
        start, R = self.pose()
        rot = R if rot is None else rot
        seed = self.f.data.qpos[:9].copy()
        for i in range(max(1, round(seconds / 0.02))):
            t = (i + 1) / max(1, round(seconds / 0.02))
            s = t**3 * (10 + t * (-15 + 6 * t))
            q, error = self.solve(start + (np.asarray(xyz) - start) * s, rot, seed)
            seed = q
            if error > 0.003:
                raise RuntimeError(f"Arm IK residual {error:.4f} m")
            self.command({self.f.names[j]: float(q[j]) for j in range(9)}, 0.02)

    def fingers(self, pose, seconds=0.5):
        m = self.m
        d = self.ik
        d.qpos[:] = self.f.data.qpos
        for name, q in pose.items():
            d.qpos[m.joint(name).qposadr[0]] = q
        mujoco.mj_forward(m, d)
        return self.command(
            {
                self.f.names[i]: float(
                    np.clip(d.actuator_length[i], *m.actuator_ctrlrange[i])
                )
                for i in range(9, m.nu)
            },
            seconds,
        )

    def pencil_contacts(self):
        return [c for c in self.f.observe()["contacts"] if "pencil" in c["bodies"]]
