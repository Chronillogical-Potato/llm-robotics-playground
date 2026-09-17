"""Development of frictional holding during writing, separate from fixed-grasp demo."""

import json
import numpy as np
from firmware import Firmware


class PhysicalFirmware(Firmware):
    def __init__(self):
        super().__init__("scene.xml", "initial_state.json")

    def reset(self):
        super().reset()
        self.targets[:] = json.loads(self.state_path.read_text())["targets"]


def relative(f):
    m, d = f.model, f.data
    p = m.body("rh_palm").id
    b = m.body("pencil").id
    R = d.xmat[p].reshape(3, 3)
    return R.T @ (d.xpos[b] - d.xpos[p]), R.T @ d.xmat[b].reshape(3, 3)[:, 2]


class PhysicalWriter:
    def __init__(self, c):
        self.c = c
        self.f = c.f
        self.R = c.pose()[1]
        self.zforce = 0.0
        self.force_history = []
        self.track = []

    def tip(self):
        return self.f.data.site_xpos[self.f.tip_site].copy()

    def goto(self, target, seconds=1.5):
        p, _ = self.c.pose()
        target = np.array(target)
        self.c.move(p + (target - self.tip()), seconds, self.R)
        for _ in range(4):
            error = target - self.tip()
            if np.linalg.norm(error) < 0.00015:
                break
            p, _ = self.c.pose()
            self.c.move(p + error, 0.2, self.R)

    def contact_force(self):
        force = 0.0
        import mujoco

        for ci, c in enumerate(self.f.data.contact):
            if {int(c.geom1), int(c.geom2)} == {self.f.tip_geom, self.f.paper_geom}:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.f.model, self.f.data, ci, f)
                force += max(0, float(f[0]))
        return force

    def draw(self, xy, speed=0.008, force_target=0.15):
        from firmware import PAPER_Z

        xy = np.asarray(xy)
        self.goto([*xy[0], PAPER_Z + 0.003], 1.2)
        grip, _ = self.c.pose()
        offset = self.tip() - grip
        grip = np.array([*xy[0], PAPER_Z + 0.0008 - 0.00005]) - offset
        self.c.move(grip, 1, self.R)
        seed = self.f.data.qpos[:9].copy()
        last = xy[0]
        for target in xy[1:]:
            n = max(1, round(np.linalg.norm(target - last) / (speed * 0.02)))
            for t in range(n):
                desired = last + (target - last) * (t + 1) / n
                force = self.contact_force()
                # Follow the observed free tip rather than assuming a rigid palm-tip offset.
                grip[:2] += (target - last) / n + np.clip(
                    0.15 * (desired - self.tip()[:2]), -0.00015, 0.00015
                )
                grip[2] += np.clip((force - force_target) * 0.000025, -0.00001, 0.00001)
                q, error = self.c.solve(grip, self.R, seed)
                seed = q
                self.c.command({self.f.names[i]: float(q[i]) for i in range(9)}, 0.02)
                self.track.append([*desired, *self.tip(), force])
                self.force_history.append(force)
            last = target
        self.goto(self.tip() + [0, 0, 0.006], 0.6)
