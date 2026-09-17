"""Geometry-guided local untangling with a direct outside wrist approach.

The right wrist stages beside the cable while the left arm approaches its
support grip. It then enters once, rather than hovering, backing out, lowering,
and returning. All executed movement is through the stock actuators.
"""

import argparse
import json
import numpy as np
from environment import ROOT, nodes, crossings
from cord_scan import scan
from strategy import solve_pose, grasp_scan
from grip import grip_observation
from control import Rollout
from validate import knot_interactions


def select(candidates, rotation=None):
    if not candidates:
        raise RuntimeError("No collision-free opposing grasp")
    return min(
        candidates,
        key=lambda c: (
            0
            if rotation is None
            else np.linalg.norm(np.asarray(c["rotation"]) - rotation),
            c["raise_by_m"],
            max([0] + [-x[1] for x in c["contacts"]]),
        ),
    )


def allow(r, side, branch, index):
    r.allowed_grasp[side] = {
        f"{branch}_cable_{i:03}" for i in range(max(0, index - 2), index + 3)
    }


def posture(r, side, point, candidate):
    r.scratch.qpos[:] = r.data.qpos
    sol = solve_pose(
        r.model,
        r.scratch,
        side,
        np.asarray(point),
        np.asarray(candidate["rotation"]),
        seed=candidate["joints"],
        forward=False,
    )
    if sol["position_error_m"] > 0.001 or sol["rotation_error_rad"] > 0.01:
        raise RuntimeError("Unreachable direct approach for " + side)
    return sol["joints"]


def open_region(r, cluster):
    branch = cluster["branch"]
    support = cluster["support_segment"]
    caught = cluster["caught_segment"]
    left = select(
        scan(r.model, r.data, branch, support, "left", spatial=True),
        r.data.site("left/gripper").xmat.reshape(3, 3).copy(),
    )
    right = select(
        scan(r.model, r.data, branch, caught, "right", spatial=True),
        r.data.site("right/gripper").xmat.reshape(3, 3).copy(),
    )
    allow(r, "left", branch, support)
    allow(r, "right", branch, caught)
    left_hover = np.asarray(left["site_point"]) + [0, 0, 0.075]
    right_outside = np.asarray(right["site_point"]) + [0.12, 0, 0.022]
    plan = {
        "cluster": cluster,
        "support_grasp": left,
        "caught_grasp": right,
        "left_hover": left_hover.tolist(),
        "right_direct_outside": right_outside.tolist(),
    }
    (ROOT / "outputs" / (r.name + "_plan.json")).write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    joints = {
        "left": posture(r, "left", left_hover, left),
        "right": posture(r, "right", right_outside, right),
    }
    duration = max(
        1.15,
        max(
            1.5 * np.max(np.abs(np.asarray(q) - r.targets[s])) / 3.0
            for s, q in joints.items()
        ),
    )
    r.move(
        "approach the support; position the right hand directly beside the loop",
        {},
        duration,
        0.1,
        joint_targets=joints,
        finger_targets={
            "left": left["finger_position"] + 0.002,
            "right": right["finger_position"] + 0.006,
        },
    )
    if r.error:
        return
    left = select(
        scan(r.model, r.data, branch, support, "left", spatial=True),
        np.asarray(left["rotation"]),
    )
    r.move(
        "take the exposed support span",
        {"left": left["site_point"]},
        0.65,
        0.1,
        rotations={"left": left["rotation"]},
        finger_targets={"left": left["finger_position"] + 0.002},
    )
    r.move(
        "secure the support",
        {},
        0.18,
        0.2,
        finger_targets={"left": left["finger_position"] - 0.001},
    )
    if r.error:
        return
    r.track_cable_grip("left", branch, support)
    r.move(
        "lift the tangled middle to expose slack",
        {"left": r.data.site("left/gripper").xpos.copy() + [-0.025, 0, 0.11]},
        0.95,
        0.2,
    )
    if r.error:
        return
    take_and_spread(r, cluster)


def take_and_spread(r, cluster):
    """Continue a held support with one direct caught-loop entry."""
    branch = cluster["branch"]
    caught = cluster["caught_segment"]
    allow(r, "right", branch, caught)
    rotation = r.data.site("right/gripper").xmat.reshape(3, 3).copy()
    right = select(
        scan(r.model, r.data, branch, caught, "right", spatial=True), rotation
    )
    r.move(
        "enter once from the open side and take the caught loop",
        {"right": right["site_point"]},
        0.6,
        0.1,
        rotations={"right": right["rotation"]},
        finger_targets={"right": right["finger_position"] + 0.006},
    )
    if r.error:
        return
    grip_and_spread(r, cluster)


def grip_and_spread(r, cluster):
    branch = cluster["branch"]
    caught = cluster["caught_segment"]
    allow(r, "right", branch, caught)
    # Track the observed wire through closure instead of allowing it to fall
    # below a fixed wrist pose while the fingers close.
    right = select(
        scan(r.model, r.data, branch, caught, "right", spatial=True),
        r.data.site("right/gripper").xmat.reshape(3, 3),
    )
    r.follow_and_pinch("right", branch, caught, right)
    if r.error:
        return
    spread(r, cluster)


def spread(r, cluster):
    previous_dt = r.model.opt.timestep
    r.model.opt.timestep = min(previous_dt, 0.0005)
    left = r.data.site("left/gripper").xpos.copy()
    right = r.data.site("right/gripper").xpos.copy()
    r.move(
        "open the local entanglement sideways",
        {"left": left + [-0.035, 0, 0], "right": right + [0.08, 0, 0]},
        0.8,
        0.15,
    )
    if r.error:
        r.model.opt.timestep = previous_dt
        return
    left = r.data.site("left/gripper").xpos.copy()
    right = r.data.site("right/gripper").xpos.copy()
    r.move(
        "elevate the separated spans",
        {"left": left + [-0.01, 0, 0.02], "right": right + [0.025, 0, 0.07]},
        0.65,
        0.15,
    )
    r.model.opt.timestep = previous_dt


def straighten(r, cluster):
    """Release the support, take its nearby free end, and draw the loose folds out."""
    if "left" in r.held:
        r.held.pop("left", None)
        r.grip_reference.pop("left", None)
        r.grip_local_points.pop("left", None)
        previous_dt = r.model.opt.timestep
        r.model.opt.timestep = min(previous_dt, 0.0005)
        r.move(
            "release the relaxed support fold",
            {},
            0.18,
            0.2,
            finger_targets={"left": 0.025},
        )
        r.model.opt.timestep = previous_dt
    if r.error:
        return
    obj = cluster["free_end"]
    cs = end_candidates(r, obj)
    if not cs:
        raise RuntimeError("No free-end grasp after relaxation")
    c = min(
        cs,
        key=lambda q: (
            np.linalg.norm(np.asarray(q["joints"]) - r.targets["left"]),
            q["raise_by_m"],
        ),
    )
    r.allowed_grasp["left"] = {obj}
    hover = np.asarray(c["site_point"]) + [0, 0, 0.075]
    joints = posture(r, "left", hover, c)
    duration = max(
        0.9, 1.5 * np.max(np.abs(np.asarray(joints) - r.targets["left"])) / 3.0
    )
    r.move(
        "move directly to the exposed free end",
        {},
        duration,
        0.1,
        joint_targets={"left": joints},
        finger_targets={"left": c["finger_position"] + 0.002},
    )
    if r.error:
        return
    grasp_free_end(r, cluster, np.asarray(c["rotation"]))


def end_candidates(r, obj):
    if "earbud" in obj:
        from stem_scan import scan_stem

        cs = scan_stem(r.model, r.data, obj)
        if cs:
            return cs
    cs = grasp_scan(r.model, r.data, [(obj, "left")], grasp_mode="stem")[obj]
    return cs


def grasp_free_end(r, cluster, rot=None):
    obj = cluster["free_end"]
    cs = end_candidates(r, obj)
    if not cs:
        raise RuntimeError("No accessible free-end grasp")
    if rot is None:
        rot = r.data.site("left/gripper").xmat.reshape(3, 3).copy()
    c = min(
        cs,
        key=lambda q: (
            np.linalg.norm(np.asarray(q["rotation"]) - rot),
            q["raise_by_m"],
        ),
    )
    r.allowed_grasp["left"] = {obj}
    duration = max(
        0.65, 1.5 * np.max(np.abs(np.asarray(c["joints"]) - r.targets["left"])) / 3.0
    )
    r.move(
        "take the free end",
        {"left": c["site_point"]},
        duration,
        0.1,
        rotations={"left": c["rotation"]},
        finger_targets={"left": c["finger_position"] + 0.002},
    )
    r.move(
        "secure the free end",
        {},
        0.18,
        0.2,
        finger_targets={
            "left": c["finger_position"] - (0.0006 if obj == "plug" else 0.00125)
        },
    )
    if r.error:
        return
    draw_free_end(r, cluster)


def draw_free_end(r, cluster):
    obj = cluster["free_end"]
    r.allowed_grasp["left"] = {obj}
    if not grip_observation(r.model, r.data, "left", obj)["opposing_contacts"]:
        r.move(
            "finish closing on the free end",
            {},
            0.18,
            0.1,
            finger_targets={"left": r.fingers["left"] - 0.001},
        )
    if r.error:
        return
    if not grip_observation(r.model, r.data, "left", obj)["opposing_contacts"]:
        raise RuntimeError("No opposing free-end contact; lift inhibited")
    r.track_hardware_grip("left", obj)
    if r.error:
        return
    left = r.data.site("left/gripper").xpos.copy()
    right = r.data.site("right/gripper").xpos.copy()
    lifted = left.copy()
    lifted[0] -= 0.045
    lifted[2] = 0.114
    r.move("lift the free end to free the remaining folds", {"left": lifted}, 1.0, 0.1)
    if r.error:
        return
    left = r.data.site("left/gripper").xpos.copy()
    right = r.data.site("right/gripper").xpos.copy()
    if cluster["id"] == "earbud":
        branch, index = r.held["right"].rsplit("_cable_", 1)
        index = int(index)
        curve = nodes(r.model, r.data)[branch]
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        arc = float(lengths[index + 1 :].sum() + lengths[index] / 2)
        earbud_goal = np.array([-0.30, -0.125, 0.20])
        cord_goal = earbud_goal + [0.95 * arc, 0, 0]
        left += earbud_goal - r.data.body(obj).xpos
        right += cord_goal - r.data.geom(f"{branch}_segment_{index:03}").xpos
    else:
        left[0] = -0.30
        left[2] = 0.114
        right[0] = 0.21
        right[2] = max(right[2], left[2])
    r.move("draw out the relaxed cable", {"left": left, "right": right}, 1.4, 0.3)


def place(r, cluster, withdraw=True):
    left = r.data.site("left/gripper").xpos.copy()
    right = r.data.site("right/gripper").xpos.copy()
    left[2] += 0.0204 - r.data.body(cluster["free_end"]).xpos[2]
    if cluster["id"] == "earbud":
        left[2] += 0.0016
        body = r.data.body(r.held["right"])
        material = body.xpos + body.xmat.reshape(3, 3) @ np.asarray(
            r.grip_local_points["right"]
        )
        right[2] += 0.028 - material[2]
        r.move(
            "lay the straightened earbud lead below the trunk",
            {"left": left, "right": right},
            1.1,
            0.2,
        )
    else:
        left[1] = 0.10
        right = np.array([0.23, 0.105, 0.033])
        # Opening the support during laydown lets the slack cable fall naturally;
        # the hardware-end grasp remains monitored until the mat supports it.
        r.held.pop("right", None)
        r.grip_reference.pop("right", None)
        r.grip_local_points.pop("right", None)
        previous_dt = r.model.opt.timestep
        r.model.opt.timestep = min(previous_dt, 0.0005)
        r.move(
            "lower the free end and open the right support",
            {"left": left, "right": right},
            1.0,
            0.2,
            finger_targets={"right": 0.025},
        )
        r.model.opt.timestep = previous_dt
    if r.error:
        return
    r.held.clear()
    r.grip_reference.clear()
    r.grip_local_points.clear()
    r.move(
        "release the headphones",
        {},
        0.18,
        0.25,
        finger_targets={"left": 0.025, "right": 0.025},
    )
    # One diagonal departure replaces the previous vertical withdrawal plus
    # a second later move to uncover the final earbud branches.
    if not r.error and cluster["id"] == "earbud" and withdraw:
        r.move(
            "withdraw both open hands directly to the sides",
            {"left": [-0.33, 0.08, 0.17], "right": [0.34, -0.13, 0.18]},
            1.0,
            0.2,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="settled")
    p.add_argument("--name", required=True)
    p.add_argument("--cluster", choices=["middle", "earbud"], default="middle")
    p.add_argument(
        "--stage",
        choices=[
            "open",
            "take_and_spread",
            "grip_and_spread",
            "straighten",
            "draw",
            "place",
            "finish",
            "lay_out",
            "take_end",
            "all",
        ],
        default="open",
    )
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--keep-hands", action="store_true")
    a = p.parse_args()
    task = json.loads((ROOT / "task.json").read_text())
    cluster = next(c for c in task["layout"]["clusters"] if c["id"] == a.cluster)
    r = Rollout(
        a.state, a.name, dt=a.dt, iterations=150, solver="PGS", jacobian="dense"
    )
    before = knot_interactions(crossings(nodes(r.model, r.data)), task)
    try:
        if a.stage in ["open", "all"]:
            open_region(r, cluster)
        if a.stage == "take_and_spread":
            take_and_spread(r, cluster)
        if a.stage == "grip_and_spread":
            grip_and_spread(r, cluster)
        if a.stage == "finish":
            spread(r, cluster)
        if not r.error and a.stage in ["straighten", "finish", "lay_out", "all"]:
            straighten(r, cluster)
        if not r.error and a.stage == "draw":
            draw_free_end(r, cluster)
        if not r.error and a.stage == "take_end":
            grasp_free_end(r, cluster)
        if not r.error and a.stage in ["place", "finish", "lay_out", "take_end", "all"]:
            place(r, cluster, withdraw=not a.keep_hands)
    except RuntimeError as e:
        r.error = str(e)
    r.write()
    path = ROOT / "outputs" / (a.name + ".json")
    report = json.loads(path.read_text())
    report["cluster_action"] = cluster["id"]
    report["cluster_structure_before"] = before
    report["cluster_structure_after"] = knot_interactions(
        crossings(nodes(r.model, r.data)), task
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cluster_before": before,
                "cluster_after": report["cluster_structure_after"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
