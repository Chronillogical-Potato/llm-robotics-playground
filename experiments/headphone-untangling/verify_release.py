"""Check a released physical checkpoint; projected crossings alone are insufficient."""

import argparse
import hashlib
import json
import mujoco
import numpy as np
from environment import ROOT, load, nodes, crossings, aloha_checks


def verify(name):
    m, d = load(name)
    robot_integrity = aloha_checks.robot_check(m)
    report = json.loads((ROOT / "outputs" / (name + ".json")).read_text())
    curves = nodes(m, d)
    raw_crossings = crossings(curves)

    # The three roots meet inside one physical splitter. Shared attachment
    # geometry is not a crossing of two free cable spans.
    def at_junction(branch, index):
        return index >= len(curves["main"]) - 3 if branch == "main" else index <= 1

    meaningful = [
        c
        for c in raw_crossings
        if not (
            c["a"][0] != c["b"][0] and at_junction(*c["a"]) and at_junction(*c["b"])
        )
    ]
    robot_contacts = []
    nonlocal_cable_contacts = []
    for c in d.contact:
        if c.dist > 0.0001:
            continue
        bodies = [m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
        if any(b.startswith(("left/", "right/")) for b in bodies) and any(
            "_cable_" in b or b in ["plug", "splitter", "left_earbud", "right_earbud"]
            for b in bodies
        ):
            robot_contacts.append({"bodies": bodies, "distance_m": float(c.dist)})
        if all("_cable_" in b for b in bodies):
            a, b = [
                (x.rsplit("_cable_", 1)[0], int(x.rsplit("_cable_", 1)[1]))
                for x in bodies
            ]
            adjacent = (a[0] == b[0] and abs(a[1] - b[1]) <= 2) or (
                a[0] != b[0] and at_junction(*a) and at_junction(*b)
            )
            if not adjacent:
                nonlocal_cable_contacts.append(
                    {"bodies": bodies, "distance_m": float(c.dist)}
                )
    track = np.load(ROOT / "outputs" / (name + "_trajectory.npz"))
    earlier = max(0, int(np.searchsorted(track["time"], track["time"][-1] - 0.25)))
    old = mujoco.MjData(m)
    old.qpos[:] = track["qpos"][earlier]
    mujoco.mj_fwdPosition(m, old)
    old_curves = nodes(m, old)
    drift = max(
        float(np.max(np.linalg.norm(p - old_curves[k], axis=1)))
        for k, p in curves.items()
    )
    max_height = max(float(p[:, 2].max()) for p in curves.values())
    all_points = np.concatenate(list(curves.values()))
    positions = {
        n: d.body(n).xpos.tolist()
        for n in ["plug", "splitter", "left_earbud", "right_earbud"]
    }
    checks = {
        "no_rollout_error": report.get("error") is None,
        "both_grips_released": not report.get("held"),
        "no_robot_headphone_contact": not robot_contacts,
        "no_free_span_crossings": not meaningful,
        "no_nonlocal_cable_contacts": not nonlocal_cable_contacts,
        "cable_laid_on_mat": max_height < 0.035,
        "cable_within_mat": bool(
            np.all(np.abs(all_points[:, :2]) < np.array([0.35, 0.265]))
        ),
        "settled_under_1mm_motion_in_last_quarter_second": drift < 0.001,
        "penetration_guard_passed": report["peak_cable_penetration_m"] <= 0.0002,
        "original_gripper_equalities_only": m.neq == 2,
    }
    result = {
        "state": name,
        "scene_sha256": hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest(),
        "untangled": all(checks.values()),
        "checks": checks,
        "parts": positions,
        "raw_projected_crossings": raw_crossings,
        "free_span_crossings": meaningful,
        "robot_headphone_contacts": robot_contacts,
        "nonlocal_cable_contacts": nonlocal_cable_contacts,
        "max_cable_height_m": max_height,
        "last_quarter_second_max_node_motion_m": drift,
        "robot_integrity": robot_integrity,
        "physical_evidence": "Final released geometry, absence of nonlocal contacts, settling and actuator-only continuous recorded dynamics. No claim of global motion optimality.",
    }
    (ROOT / "outputs" / (name + "_verification.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    a = p.parse_args()
    verify(a.state)
