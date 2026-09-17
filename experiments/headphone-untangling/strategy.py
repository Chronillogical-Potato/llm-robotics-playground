"""Ground-truth crossing analysis and static stock-gripper grasp candidates.

This is planning, not a manipulation rollout. Joint assignments in the grasp
scan are hypothetical configurations, never presented as executed movements.
"""

import argparse
import hashlib
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from environment import ROOT, load, nodes, aloha_checks, geometry_contacts


def endpoint_analysis(model, data):
    curves = nodes(model, data)
    arcs = {
        k: np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        for k, p in curves.items()
    }
    segments = [
        (name, i, a, b)
        for name, p in curves.items()
        for i, (a, b) in enumerate(zip(p[:-1], p[1:]))
    ]
    events = {name: [] for name in curves}
    for k, (na, ia, a, b) in enumerate(segments):
        for nb, ib, c, d in segments[k + 1 :]:
            if na == nb and abs(ia - ib) <= 2:
                continue
            matrix = np.column_stack([(b - a)[:2], -(d - c)[:2]])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            u, v = np.linalg.solve(matrix, (c - a)[:2])
            if not (0 < u < 1 and 0 < v < 1):
                continue
            pa, pb = a + u * (b - a), c + v * (d - c)
            for n, i, t, p, other, oi, over in [
                (na, ia, u, pa, nb, ib, pa[2] > pb[2]),
                (nb, ib, v, pb, na, ia, pb[2] > pa[2]),
            ]:
                s = arcs[n][i] + t * (arcs[n][i + 1] - arcs[n][i])
                events[n].append(
                    {
                        "arc_m": float(s),
                        "segment": i,
                        "other": [other, oi],
                        "over": bool(over),
                        "position": p.tolist(),
                        "height_gap_m": float(abs(pa[2] - pb[2])),
                    }
                )
    result = {}
    for name, branch in [
        ("plug", "main"),
        ("left_earbud", "left"),
        ("right_earbud", "right"),
    ]:
        ordered = sorted(
            events[branch], key=lambda e: e["arc_m"], reverse=branch != "main"
        )
        first_under = next(
            (i for i, e in enumerate(ordered) if not e["over"]), len(ordered)
        )
        prefix_length = (
            (
                ordered[first_under]["arc_m"]
                if branch == "main"
                else arcs[branch][-1] - ordered[first_under]["arc_m"]
            )
            if first_under < len(ordered)
            else arcs[branch][-1]
        )
        result[name] = {
            "branch": branch,
            "top_crossings_before_first_underpass": first_under,
            "cable_length_to_first_underpass_m": float(prefix_length),
            "ordered_crossings": ordered,
        }
    return result


def fingertip_center(model, data, side):
    site = data.site(side + "/gripper")
    points = [
        data.geom(f"{side}/{finger}_g{i}").xpos
        for finger in ["left", "right"]
        for i in range(3)
    ]
    return site.xmat.reshape(3, 3).T @ (np.mean(points, axis=0) - site.xpos)


def solve_pose(model, data, side, point, rotation, seed=None, forward=True):
    ids = [model.joint(side + "/" + j).id for j in aloha_checks.JOINTS]
    adr = model.jnt_qposadr[ids]
    lower, upper = model.jnt_range[ids].T
    site = model.site(side + "/gripper").id
    start = data.qpos[adr].copy() if seed is None else np.array(seed)

    def residual(q):
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        actual = data.site_xmat[site].reshape(3, 3)
        return np.r_[
            data.site_xpos[site] - point,
            0.1 * Rotation.from_matrix(rotation @ actual.T).as_rotvec(),
        ]

    candidates = (
        [start, [0, -0.5, 0.6, 0, 1.3, 0], [0, 0.1, -0.8, 0, 2, 0]]
        if seed is None
        else [start]
    )
    best = None
    for q in candidates:
        result = least_squares(
            residual,
            np.clip(q, lower + 1e-7, upper - 1e-7),
            bounds=(lower, upper),
            max_nfev=100,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        score = np.linalg.norm(error)
        if best is None or score < best[0]:
            best = (score, result.x.copy(), error.copy())
        if score < 1e-6:
            break
    data.qpos[adr] = best[1]
    if forward:
        mujoco.mj_forward(model, data)
    else:
        mujoco.mj_kinematics(model, data)
    return {
        "joints": best[1].tolist(),
        "position_error_m": float(np.linalg.norm(best[2][:3])),
        "rotation_error_rad": float(np.linalg.norm(best[2][3:]) / 0.1),
    }


def grasp_contacts(model, data, side, object_name):
    good, bad = [], []
    for c in data.contact:
        if c.dist > 0.0001:
            continue
        ids = [int(c.geom1), int(c.geom2)]
        bodies = [model.body(model.geom_bodyid[g]).name for g in ids]
        if not any(b.startswith(side + "/") for b in bodies):
            continue
        item = {
            "bodies": bodies,
            "geoms": [model.geom(g).name or f"geom_{g}" for g in ids],
            "distance_m": float(c.dist),
        }
        if object_name in bodies and any(
            b in [side + "/left_finger_link", side + "/right_finger_link"]
            for b in bodies
        ):
            good.append(item)
        elif c.dist < -0.00005:
            bad.append(item)
    return good, bad


def grasp_scan(model, data, objects=None, grasp_mode="housing"):
    initial_qpos = data.qpos.copy()
    results = {}
    for object_name, side in objects or [
        ("plug", "left"),
        ("right_earbud", "right"),
        ("left_earbud", "left"),
    ]:
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
        body = data.body(object_name)
        rbody = body.xmat.reshape(3, 3).copy()
        target = data.site(
            "plug_grasp" if object_name == "plug" else object_name + "_grasp"
        ).xpos.copy()
        if object_name != "plug":
            target = (
                body.xpos + rbody[:, 0] * 0.007
                if grasp_mode == "stem"
                else target - rbody[:, 1] * 0.0007
            )
        yaw0 = np.arctan2(rbody[1, 0], rbody[0, 0])
        offset = fingertip_center(model, data, side)
        choices = []
        for angle in (
            [0, np.pi] if grasp_mode == "stem" else [0, np.pi / 2, np.pi, -np.pi / 2]
        ):
            rotation = (
                Rotation.from_euler("z", yaw0 + angle).as_matrix()
                @ Rotation.from_euler("y", np.pi / 2).as_matrix()
            )
            for raise_by in (
                [0.0005, 0.001, 0.0015]
                if grasp_mode == "stem"
                else [0.0015, 0.003]
                if object_name == "plug"
                else [0.001, 0.003]
            ):
                data.qpos[:] = initial_qpos
                point = target + [0, 0, raise_by] - rotation @ offset
                solution = solve_pose(model, data, side, point, rotation, forward=False)
                if (
                    solution["position_error_m"] > 0.0005
                    or solution["rotation_error_rad"] > 0.005
                ):
                    continue
                for opening in np.arange(0.023, 0.008, -0.00025):
                    for finger in ["left_finger", "right_finger"]:
                        data.qpos[model.joint(side + "/" + finger).qposadr[0]] = opening
                    geometry_contacts(model, data)
                    good, bad = grasp_contacts(model, data, side, object_name)
                    touching = {
                        b for g in good for b in g["bodies"] if b.startswith(side + "/")
                    }
                    if len(touching) == 2 and not bad:
                        deepest = max([0] + [-g["distance_m"] for g in good])
                        if deepest < 0.001:
                            choices.append(
                                {
                                    "arm": side,
                                    "target_body": object_name,
                                    "site_point": point.tolist(),
                                    "grasp_mode": grasp_mode,
                                    "rotation": rotation.tolist(),
                                    "pad_center_offset": offset.tolist(),
                                    "finger_position": float(opening),
                                    "raise_by_m": raise_by,
                                    "contacts": good,
                                    "max_static_overlap_m": float(deepest),
                                    **solution,
                                }
                            )
                        break
                    if bad:
                        break
        results[object_name] = sorted(
            choices, key=lambda c: (-len(c["contacts"]), c["max_static_overlap_m"])
        )
        print(
            object_name, "static opposing-contact candidates", len(choices), flush=True
        )
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    return results
