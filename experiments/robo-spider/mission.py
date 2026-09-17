"""Scripted pilot: world waypoints and Cartesian hand commands, no joint access."""

import numpy as np, math

DOWN = np.diag([1.0, -1.0, -1.0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission:
    def __init__(self, fw):
        self.fw = fw
        self.stage = "APPROACH"
        self.since = 0
        self.done = False
        self.carry = None
        self.history = []
        self.nav_key = None
        self.arrived = False
        self.ready_since = None

    def enter(self, stage, t):
        self.stage = stage
        self.since = t
        self.ready_since = None
        self.history.append({"time": float(t), "stage": stage})

    def confirmed(self, condition, dwell=0.2):
        t = self.fw.d.time
        if not condition:
            self.ready_since = None
            return False
        if self.ready_since is None:
            self.ready_since = t
        return t - self.ready_since >= dwell

    def settled(self):
        f = self.fw
        v = f.d.qvel[f.m.joint("root").dofadr[0] : f.m.joint("root").dofadr[0] + 6]
        return self.confirmed(
            abs(f.speed) < 0.002
            and abs(f.turn) < 0.005
            and np.linalg.norm(v[:3]) < 0.015
            and np.linalg.norm(v[3:]) < 0.04
        )

    def hands_ready(self):
        f = self.fw
        ready = True
        for side, h in f.hands.items():
            if h["goal"] is None:
                return False
            p, R = h["goal"]
            actual, A = f.hand_pose(side)
            if h["frame"] == "body":
                b = f.d.body("robot")
                B = b.xmat.reshape(3, 3)
                p = b.xpos + B @ p
                R = B @ R
            start, duration, _, _ = h["trajectory"]
            ready &= (
                f.d.time >= start + duration
                and np.linalg.norm(actual - p) < 0.012
                and np.linalg.norm(A - R) < 0.12
            )
        return self.confirmed(ready)

    def grippers_ready(self, closing):
        f = self.fw
        ready = True
        for side, name in [("right", "target_mug"), ("left", "target_block")]:
            aid = f.m.actuator(side + "_grip_fingers_actuator").id
            ready &= abs(f.d.ctrl[aid] - (255 if closing else 0)) < 2
            bid = f.m.body(name).id
            touch = False
            for c in f.d.contact:
                b1, b2 = f.m.geom_bodyid[c.geom1], f.m.geom_bodyid[c.geom2]
                if bid in (b1, b2):
                    other = b2 if b1 == bid else b1
                    touch |= f.m.body(other).name.startswith(side + "_grip_")
            ready &= touch if closing else not touch
        return self.confirmed(ready, 0.3)

    def nav(self, target, reverse=False, heading=None, cruise=0.30, arrival=0.075):
        f = self.fw
        d = f.d
        p = d.body("robot").xpos
        R = d.body("robot").xmat.reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        key = (tuple(target), reverse, heading)
        if key != self.nav_key:
            self.nav_key = key
            self.arrived = False
        delta = np.array(target) - p[:2]
        dist = np.linalg.norm(delta)
        if dist < arrival:
            self.arrived = True
        desired = math.atan2(delta[1], delta[0]) + (math.pi if reverse else 0)
        err = wrap(desired - yaw)
        if self.arrived:
            err = wrap((yaw if heading is None else heading) - yaw)
            f.command_drive(0, float(np.clip(err * 0.8, -0.25, 0.25)))
            return abs(err) < 0.055
        speed = (
            min(cruise, dist * 0.45) * max(0, math.cos(err)) ** 2
            if abs(err) < 1.45
            else 0
        )
        f.command_drive(
            float(-speed if reverse else speed), float(np.clip(err * 0.8, -0.28, 0.28))
        )
        return False

    def hands(self, right, left, grip, rotation=DOWN, frame="world"):
        self.fw.command_hand("right", right, rotation, grip, frame)
        self.fw.command_hand("left", left, rotation, grip, frame)

    def step(self):
        f = self.fw
        t = f.d.time
        e = t - self.since
        s = self.stage
        if s == "APPROACH":
            if self.nav([-0.15, 0.40], heading=0):
                f.command_drive()
                self.enter("DOCK_SOURCE", t)
        elif s == "DOCK_SOURCE":
            if self.nav([0.39, 0.40], heading=0, cruise=0.13):
                f.command_drive()
                self.enter("SETTLE", t)
        elif s == "SETTLE":
            f.command_drive()
            if self.settled():
                self.enter("REACH", t)
        elif s == "REACH":
            self.hands([1.15, 0.12, 0.81], [1.14, 0.69, 0.82], 0)
            if self.hands_ready():
                self.enter("LOWER", t)
        elif s == "LOWER":
            self.hands([1.15, 0.12, 0.642], [1.14, 0.69, 0.654], 0)
            if self.hands_ready():
                self.enter("GRASP", t)
        elif s == "GRASP":
            self.hands([1.15, 0.12, 0.642], [1.14, 0.69, 0.654], 1)
            if self.grippers_ready(True):
                self.enter("LIFT", t)
        elif s == "LIFT":
            self.hands([0.92, 0.10, 0.85], [0.92, 0.69, 0.85], 1)
            if self.hands_ready():
                R = f.d.body("robot").xmat.reshape(3, 3)
                p = f.d.body("robot").xpos
                self.carry = {
                    side: (R.T @ (f.hand_pose(side)[0] - p), R.T @ f.hand_pose(side)[1])
                    for side in ["right", "left"]
                }
                for side, (pos, rot) in self.carry.items():
                    f.command_hand(side, pos, rot, 1, "body")
                self.enter("BACK_AWAY", t)
        elif s == "BACK_AWAY":
            if self.nav([-1.65, 0.40], reverse=True):
                self.enter("AROUND_BARRIER", t)
        elif s == "AROUND_BARRIER":
            if self.nav([-1.32, 1.25], heading=math.pi / 2, cruise=0.14, arrival=0.12):
                f.command_drive()
                self.enter("DOCK_DEST", t)
        elif s == "DOCK_DEST":
            if self.nav([-1.32, 1.78], heading=math.pi / 2, cruise=0.13):
                f.command_drive()
                self.enter("DEST_SETTLE", t)
        elif s == "ALIGN_DESTINATION":
            if self.nav([-1.10, 1.70], heading=math.pi / 2):
                f.command_drive()
                self.enter("DEST_SETTLE", t)
        elif s == "DEST_SETTLE":
            if self.settled():
                self.enter("PLACE_APPROACH", t)
        elif s == "PLACE_APPROACH":
            R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ DOWN
            self.hands([-0.82, 2.48, 0.84], [-1.38, 2.48, 0.84], 1, R)
            if self.hands_ready():
                self.enter("PLACE_LOWER", t)
        elif s == "PLACE_LOWER":
            R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ DOWN
            self.hands([-0.82, 2.48, 0.643], [-1.38, 2.48, 0.66], 1, R)
            if self.hands_ready():
                self.enter("RELEASE", t)
        elif s == "RELEASE":
            for h in f.hands.values():
                h["grip"] = 0
            if self.grippers_ready(False):
                self.enter("WITHDRAW_HANDS", t)
        elif s == "WITHDRAW_HANDS":
            R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ DOWN
            self.hands([-0.82, 2.40, 0.85], [-1.38, 2.40, 0.85], 0, R)
            if self.hands_ready():
                self.done = True
