"""Feedback writing with a floating G1 and a freely moving, friction-held marker."""

from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
from scipy.optimize import least_squares


class Writer:
    def tip(self):
        return self.d.site_xpos[self.tipid].copy()

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.ik = mujoco.MjData(self.m)
        m, d = self.m, self.d
        a = np.load(ROOT / "initial_grasp.npz")
        d.qpos[:] = a["qpos"]
        self.targets = a["targets"].copy()
        self.jids = m.actuator_trnid[:, 0]
        self.qa = m.jnt_qposadr[self.jids]
        self.va = m.jnt_dofadr[self.jids]
        self.arm = np.arange(29, 36)
        self.tipid = m.site("tip").id
        self.virtual = m.site("tool_target").id
        self.axissite = m.site("tool_axis").id
        self.palm = m.body("right_wrist_yaw_link").id
        self.marker = m.body("marker").id
        self.board = m.geom("whiteboard").id
        self.tipgeom = m.geom("marker_tip").id
        self.barrel = m.geom("marker_barrel").id
        # Place the same settled grasp farther down the marker barrel, before the episode starts.
        mujoco.mj_forward(m, d)
        d.qpos[50:53] += d.xmat[self.marker].reshape(3, 3)[:, 2] * 0.050
        mujoco.mj_forward(m, d)
        self.force = 0
        self.filtered_force = 0
        self.marks = []
        self.peak = np.zeros(m.nu)
        self.frames = []
        self.track = []
        self.phase = "Ready"
        self.tick = 0
        self.mincontacts = 100
        self.maxslip = 0
        self.minheight = 1.0
        self.maxforce = 0.0
        self.nontip_board_contacts = 0
        self.effective_peak = np.zeros(m.nu)
        rel, rot = self.relative()
        q, e = self.solve([0.40, -0.10, 1.10], robust=True)
        self.targets[self.arm] = q
        d.qpos[self.qa[self.arm]] = q
        mujoco.mj_forward(m, d)
        R = d.xmat[self.palm].reshape(3, 3)
        d.qpos[50:53] = d.xpos[self.palm] + R @ rel
        mujoco.mju_mat2Quat(d.qpos[53:57], (R @ rot).ravel())
        mujoco.mj_forward(m, d)
        self.initialrel = self.relative()[0]

    def relative(self):
        d = self.d
        R = d.xmat[self.palm].reshape(3, 3)
        return R.T @ (d.xpos[self.marker] - d.xpos[self.palm]), R.T @ d.xmat[
            self.marker
        ].reshape(3, 3)

    def sync_tool(self):
        m, d = self.m, self.d
        R = d.xmat[self.palm].reshape(3, 3)
        m.site_pos[self.virtual] = R.T @ (self.tip() - d.xpos[self.palm])
        mujoco.mju_mat2Quat(
            m.site_quat[self.axissite],
            (R.T @ d.xmat[self.marker].reshape(3, 3)).ravel(),
        )

    def solve(self, target, axis=None, seed=None, robust=False):
        m, k = self.m, self.ik
        self.sync_tool()
        k.qpos[:] = self.d.qpos
        cols = self.qa[self.arm]
        vcols = self.va[self.arm]
        lo = m.jnt_range[self.jids[self.arm], 0] + 0.003
        hi = m.jnt_range[self.jids[self.arm], 1] - 0.003
        hi[1] = -0.03
        lo[0] = max(lo[0], -1.8)
        lo[2] = max(lo[2], -1.6)
        hi[2] = min(hi[2], 1.6)
        lo[4] = 0.6
        axis = np.array([1.0, 0, 0]) if axis is None else np.array(axis)
        reference = self.targets[self.arm].copy() if seed is None else seed.copy()
        regularization = 0.00001 if robust else 0.005

        def fun(q):
            k.qpos[cols] = q
            mujoco.mj_kinematics(m, k)
            return np.r_[
                k.site_xpos[self.virtual] - target,
                0.12 * (k.site_xmat[self.axissite].reshape(3, 3)[:, 2] - axis),
                regularization * (q - reference),
            ]

        def jac(q):
            k.qpos[cols] = q
            mujoco.mj_kinematics(m, k)
            mujoco.mj_comPos(m, k)
            jp = np.zeros((3, m.nv))
            jr = jp.copy()
            mujoco.mj_jacSite(m, k, jp, jr, self.virtual)
            a = k.site_xmat[self.axissite].reshape(3, 3)[:, 2]
            skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            return np.vstack(
                [jp[:, vcols], -0.12 * skew @ jr[:, vcols], regularization * np.eye(7)]
            )

        q = self.targets[self.arm] if seed is None else seed
        res = least_squares(
            fun,
            np.clip(q, lo, hi),
            jac=jac,
            bounds=(lo, hi),
            max_nfev=60 if robust else 12,
            ftol=1e-6,
            xtol=1e-7,
            gtol=1e-7,
        )
        if robust and np.linalg.norm(res.fun[:6]) > 0.001:
            rng = np.random.default_rng(19)
            for _ in range(15):
                trial = least_squares(
                    fun, rng.uniform(lo, hi), jac=jac, bounds=(lo, hi), max_nfev=100
                )
                if np.linalg.norm(trial.fun) < np.linalg.norm(res.fun):
                    res = trial
                if np.linalg.norm(res.fun[:6]) < 1e-5:
                    break
        return (
            res.x if robust else reference + np.clip(res.x - reference, -0.018, 0.018)
        ), float(np.linalg.norm(res.fun[:3]))

    def step(self):
        m, d = self.m, self.d
        ff = np.zeros(m.nu)
        ff[:36] = d.qfrc_bias[self.va[:36]] / m.actuator_gainprm[:36, 0]
        d.ctrl[:] = np.clip(
            self.targets + ff, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(m, d)
        self.peak = np.maximum(self.peak, np.abs(d.actuator_force))
        self.effective_peak = np.maximum(
            self.effective_peak, np.abs(d.qfrc_actuator[self.va])
        )
        self.minheight = min(self.minheight, float(d.qpos[2]))
        self.force = 0
        hands = set()
        for i, c in enumerate(d.contact):
            gs = {int(c.geom1), int(c.geom2)}
            if self.board in gs and self.tipgeom not in gs:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                if f[0] > 0.01:
                    self.nontip_board_contacts += 1
            if self.tipgeom in gs or self.barrel in gs:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                if gs == {self.board, self.tipgeom}:
                    self.force += max(0, f[0])
                    if f[0] > 0.015:
                        p = np.array([0.4298, c.pos[1], c.pos[2]])
                        if (
                            not self.marks
                            or np.linalg.norm(p - self.marks[-1][:3]) > 0.0004
                        ):
                            self.marks.append(np.r_[p, f[0]])
                if self.barrel in gs and f[0] > 0.01:
                    g = c.geom2 if c.geom1 == self.barrel else c.geom1
                    name = m.body(m.geom_bodyid[g]).name
                    if name.startswith("right_hand"):
                        hands.add(name)
        self.maxforce = max(self.maxforce, self.force)
        self.filtered_force = 0.95 * self.filtered_force + 0.05 * self.force
        self.tick += 1
        self.mincontacts = min(self.mincontacts, len(hands))
        self.maxslip = max(
            self.maxslip, np.linalg.norm(self.relative()[0] - self.initialrel)
        )
        if self.tick % 20 == 0:
            self.frames.append((d.time, d.qpos.copy(), len(self.marks), self.phase))
        if self.tick % 50 == 0 and (
            d.qpos[2] < 0.70
            or np.linalg.norm(self.relative()[0] - self.initialrel) > 0.04
        ):
            raise RuntimeError(f"Lost stance or grip at {d.time:.2f}s")
        return self.force

    def goto(self, target, duration=1.0):
        start = self.tip()
        target = np.array(target)
        duration = max(duration, np.linalg.norm(target - start) / 0.12)
        n = max(1, round(duration / 0.01))
        for i in range(n):
            s = (i + 1) / n
            s = s * s * s * (10 + s * (-15 + 6 * s))
            goal = start + (target - start) * s
            q, e = self.solve(goal)
            self.targets[self.arm] = q
            for _ in range(10):
                self.step()

    def draw(self, points, speed=0.020):
        points = np.array(points)
        self.phase = "Move"
        self.goto([0.414, *points[0]], 0.45)
        self.phase = "Writing"
        self.goto([0.4278, *points[0]], 0.4)
        depth = 0.4280
        for _ in range(70):
            depth += np.clip((0.5 - self.filtered_force) * 0.000035, -0.00003, 0.000025)
            q, e = self.solve([depth, *points[0]])
            self.targets[self.arm] = q
            for _ in range(10):
                self.step()
        last = points[0]
        for dest in points[1:]:
            n = max(1, int(np.ceil(np.linalg.norm(dest - last) / (speed * 0.01))))
            i = 0
            wait = 0
            while i < n:
                yz = last + (dest - last) * (i + 1) / n
                depth += np.clip(
                    (0.5 - self.filtered_force) * 0.000035, -0.00004, 0.000025
                )
                depth = np.clip(depth, 0.423, 0.435)
                tip = self.tip()
                goal = np.r_[depth, yz + np.clip(0.25 * (yz - tip[1:]), -0.001, 0.001)]
                q, e = self.solve(goal)
                self.targets[self.arm] = q
                loaded = 0
                for _ in range(10):
                    self.step()
                    loaded += self.force > 0.015
                if loaded >= 3:
                    i += 1
                    wait = 0
                else:
                    wait += 1
                    if wait > 160:
                        raise RuntimeError(
                            f"Unable to maintain ink contact at {self.d.time:.2f}s"
                        )
                self.track.append([self.d.time, *yz, *self.tip(), self.force, e])
            last = dest
        self.phase = "Lift"
        self.goto([0.414, *points[-1]], 0.23)

    def save(self):
        np.savez_compressed(
            OUT / "episode.npz",
            times=np.array([f[0] for f in self.frames]),
            qpos=np.array([f[1] for f in self.frames]),
            mark_counts=np.array([f[2] for f in self.frames]),
            phases=np.array([f[3] for f in self.frames]),
            marks=np.array(self.marks),
            track=np.array(self.track),
        )
