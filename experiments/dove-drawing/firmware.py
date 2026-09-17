"""Fixed low-level interface. No IK, grasp planner, drawing path, or model calls."""

from pathlib import Path
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
PAPER_Z = 0.764


class Firmware:
    def __init__(self, scene="scene.xml", state="initial_state.json"):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / scene))
        self.data = mujoco.MjData(self.model)
        self.state_path = ROOT / state
        self.tip_geom = self.model.geom("graphite_tip").id
        self.paper_geom = self.model.geom("paper").id
        self.tip_site = self.model.site("tip").id
        self.names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.targets = np.zeros(self.model.nu)
        self.marks = []
        self.last_contact = None
        self.last_deposit = None
        self.max_force = 0.0
        self.contact_samples = 0
        self.peak_torque = np.zeros(self.model.nu)
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset the selected scene to its disclosed starting state."""
        mujoco.mj_resetData(self.model, self.data)
        state = json.loads(self.state_path.read_text())
        self.data.qpos[:] = state["qpos"]
        mujoco.mj_forward(self.model, self.data)
        self.targets[:] = state.get("targets", self.data.actuator_length)
        self.marks = []
        self.last_contact = None
        self.last_deposit = None
        self.max_force = 0.0
        self.contact_samples = 0
        self.peak_torque[:] = 0

    def command(self, targets, duration_s=0.1):
        """set_targets: named joint/tendon angles, advanced for bounded simulated time.

        Joint targets are radians; FFJ0/MFJ0/RFJ0/LFJ0 are summed tendon angles.
        Every actuator stays force-limited. The caller coordinates all 27 targets.
        """
        if not 0.002 <= duration_s <= 2:
            raise ValueError("duration_s must be 0.002–2")
        goal = self.targets.copy()
        for name, value in targets.items():
            if name not in self.names:
                raise ValueError(f"Unknown actuator: {name}")
            i = self.names.index(name)
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("Nonfinite target")
            bounds = (
                self.model.actuator_ctrlrange[i]
                if self.model.actuator_ctrllimited[i]
                else self.model.jnt_range[self.model.actuator_trnid[i, 0]]
            )
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Target outside limit: {name}")
            goal[i] = value
        start = self.targets.copy()
        n = round(duration_s / self.model.opt.timestep)
        for i in range(n):
            t = (i + 1) / n
            s = t * t * t * (10 + t * (-15 + 6 * t))
            self.targets[:] = start + (goal - start) * s
            self.step()
        return self.observe()

    def step(self):
        m, d = self.model, self.data
        # Bias compensation only for direct arm/wrist actuators; actuator limits bound it.
        ff = np.zeros(m.nu)
        for i in range(9):
            j = int(m.actuator_trnid[i, 0])
            ff[i] = d.qfrc_bias[m.jnt_dofadr[j]] / m.actuator_gainprm[i, 0]
        d.ctrl[:] = self.targets + ff
        mujoco.mj_step(m, d)
        self.peak_torque = np.maximum(self.peak_torque, np.abs(d.actuator_force))
        force = 0.0
        point = None
        for ci, c in enumerate(d.contact):
            if {int(c.geom1), int(c.geom2)} == {self.tip_geom, self.paper_geom}:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                force += max(0, f[0])
                point = c.pos.copy()
        if point is not None and force > 0.015:
            self.contact_samples += 1
            self.max_force = max(self.max_force, force)
            p = np.array([point[0], point[1], PAPER_Z + 0.00008])
            # Deposit a graphite footprint only at a measured loaded contact point.
            # Spacing is independent of chatter; no line bridges an airborne interval.
            if (
                self.last_deposit is None
                or np.linalg.norm(p - self.last_deposit) > 0.00009
            ):
                self.marks.append((p.copy(), p.copy(), min(force, 2.0)))
                self.last_deposit = p
            self.last_contact = p
        else:
            self.last_contact = None
        return force

    def observe(self):
        m, d = self.model, self.data
        contacts = []
        for ci, c in enumerate(d.contact):
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f)
            if f[0] > 0.001:
                contacts.append(
                    dict(
                        bodies=[
                            m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]
                        ],
                        normal_force_N=float(f[0]),
                        point_m=c.pos.tolist(),
                    )
                )
        return dict(
            time_s=d.time,
            joint_angles_rad={
                m.joint(i).name: float(d.qpos[m.jnt_qposadr[i]]) for i in range(31)
            },
            joint_velocities_rad_s=d.qvel[:31].tolist(),
            actuator_forces=d.actuator_force.tolist(),
            tip_m=d.site_xpos[self.tip_site].tolist(),
            pencil_pose=np.r_[
                d.xpos[m.body("pencil").id], d.xquat[m.body("pencil").id]
            ].tolist(),
            contacts=contacts,
            mark_segments=len(self.marks),
            peak_tip_force_N=self.max_force,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene.xml")
    parser.add_argument("--state", default="initial_state.json")
    args = parser.parse_args()
    f = Firmware(args.scene, args.state)
    f.reset()
    print(json.dumps({"ready": True, "actuators": f.names}), flush=True)
    import sys

    for line in sys.stdin:
        try:
            c = json.loads(line)
            op = c.get("op")
            if op == "observe":
                out = f.observe()
            elif op == "reset":
                f.reset()
                out = f.observe()
            elif op == "set_targets":
                out = f.command(c["targets"], c.get("duration_s", 0.1))
            else:
                raise ValueError("op must be observe, reset, or set_targets")
            print(json.dumps(out), flush=True)
        except (ValueError, KeyError, TypeError) as e:
            print(json.dumps({"error": str(e)}), flush=True)
