"""Environment checks, including gravity settling; no untangling controller."""

import argparse
import hashlib
import json
import time
import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from environment import ROOT, aloha_checks, crossings, load, nodes, observe, save


def contact_summary(model, data):
    hits = []
    for c in data.contact:
        if c.dist >= -1e-6:
            continue
        hits.append(
            {
                "geoms": [
                    model.geom(g).name or f"geom_{g}" for g in [c.geom1, c.geom2]
                ],
                "depth_m": float(-c.dist),
            }
        )
    return sorted(hits, key=lambda h: -h["depth_m"])


def knot_interactions(items, task):
    """Classify knot, caught-bight and incidental projected crossings."""
    layout = task.get("layout", {})
    features = layout.get("entanglements", layout.get("knots", []))
    intervals = [k.get("segment_range", k.get("main_segment_range")) for k in features]

    def core(segment):
        return next(
            (
                i
                for i, (a, b) in enumerate(intervals)
                if segment[0] == features[i].get("branch", "main")
                and a <= segment[1] <= b
            ),
            None,
        )

    within = [0] * len(features)
    between = []
    outside = 0
    pairs = {}
    for c in items:
        a, b = core(c["a"]), core(c["b"])
        if a is None or b is None:
            outside += 1
            continue
        if a == b:
            within[a] += 1
        else:
            key = ":".join(str(features[i]["id"]) for i in sorted([a, b]))
            signed = c["height_difference_m"] * (1 if a < b else -1)
            pairs.setdefault(key, []).append(signed)
            between.append(signed)
    return {
        "within_each_core": within,
        "between_cores": len(between),
        "outside_features": outside,
        "feature_kinds": [f.get("kind", "overhand_knot") for f in features],
        "first_core_over_second": sum(z > 0 for z in between),
        "first_core_under_second": sum(z < 0 for z in between),
        "is_topological_proof": False,
        "between_pairs": {
            k: {
                "crossings": len(v),
                "over": sum(z > 0 for z in v),
                "under": sum(z < 0 for z in v),
            }
            for k, v in pairs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.4)
    parser.add_argument("--skip-reach", action="store_true")
    args = parser.parse_args()
    model, data = load()
    save(data, "initial")
    task = json.loads((ROOT / "task.json").read_text())
    robot = aloha_checks.robot_check(model)
    report = {
        "robot": robot,
        "mujoco_version": mujoco.__version__,
        "scene_sha256": hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest(),
        "model": {
            "bodies": model.nbody,
            "dofs": model.nv,
            "actuators": model.nu,
            "headphone_mass_kg": float(model.body_subtreemass[model.body("plug").id]),
            "free_joints": int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
            "equality_constraints": model.neq,
            "mocap_bodies": model.nmocap,
        },
        "initial_contacts": contact_summary(model, data),
        "initial_crossings": crossings(nodes(model, data)),
        "reach": {},
    }
    assert model.nu == 14 and model.nmocap == 0
    assert not report["initial_contacts"], "Reset contains penetrations"
    expected = task.get("layout", {}).get("expected_projected_crossings")
    if expected:
        structure = knot_interactions(report["initial_crossings"], task)
        assert structure["within_each_core"] == expected["within_each_feature"], (
            structure
        )
        assert structure["between_cores"] == expected["between_features"], structure
        assert structure["outside_features"] == expected["outside_features"], structure
        expected_pairs = expected.get("between_pairs", {"1:2": 2})
        assert {
            k: v["crossings"] for k, v in structure["between_pairs"].items()
        } == expected_pairs, structure
        assert all(
            v["over"] == v["under"] == 1 for v in structure["between_pairs"].values()
        ), structure
    # Only the two upstream gripper-joint equalities are present.
    assert model.neq == 2
    assert report["model"]["free_joints"] == 1
    for name, info in task["headphones"]["cables"].items():
        for i in range(info["segments"]):
            g = model.geom(f"{name}_segment_{i:03}")
            assert g.contype[0] == 1 and g.conaffinity[0] == 1
            assert abs(g.size[0] - info["diameter_m"] / 2) < 1e-12
    if not args.skip_reach:
        knot_groups = []
        for hit in report["initial_crossings"]:
            if hit["a"][0] == hit["b"][0] == "main":
                knot_groups.append(hit["xy_m"])
        knot_groups = np.array(knot_groups)
        loop_centers = (
            [
                k["center_xy_m"]
                for k in task["layout"].get(
                    "entanglements", task["layout"].get("knots", [])
                )
            ]
            if "layout" in task
            else [
                knot_groups[knot_groups[:, 0] < 0].mean(axis=0),
                knot_groups[knot_groups[:, 0] >= 0].mean(axis=0),
            ]
        )
        targets = {
            "left_earbud_approach": (
                "left",
                data.site("left_earbud_grasp").xpos.copy(),
            ),
            "right_earbud_approach": (
                "right",
                data.site("right_earbud_grasp").xpos.copy(),
            ),
            "plug_approach": ("left", data.site("plug_grasp").xpos.copy()),
            "loop_left_approach": ("left", np.r_[loop_centers[0], 0.029]),
            "loop_right_approach": ("right", np.r_[loop_centers[1], 0.029]),
        }
        for i, center in enumerate(loop_centers[2:], start=2):
            targets[f"additional_feature_{i + 1}_approach"] = (
                "left" if i % 2 == 0 else "right",
                np.r_[center, 0.029],
            )
        for name, (side, point) in targets.items():
            point += [0, 0, 0.045]
            result = aloha_checks.reach(model, side, point)
            report["reach"][name] = {"point": point.tolist(), **result}
            print(
                name,
                "error",
                round(result["position_error_m"], 7),
                "collisions",
                len(result["contacts"]),
                flush=True,
            )
    initial_nodes = nodes(model, data)
    rest_lengths = {
        k: np.linalg.norm(np.diff(v, axis=0), axis=1) for k, v in initial_nodes.items()
    }
    trajectory = [(data.time, data.qpos.copy(), data.qvel.copy())]
    peak = 0.0
    peak_self = 0.0
    self_contact_samples = 0
    start = time.monotonic()
    for i in range(int(round(args.seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            for hit in contact_summary(model, data):
                peak = max(peak, hit["depth_m"])
                if all("_segment_" in g for g in hit["geoms"]):
                    peak_self = max(peak_self, hit["depth_m"])
                    self_contact_samples += 1
        if i % 25 == 0:
            trajectory.append((data.time, data.qpos.copy(), data.qvel.copy()))
        if i % 400 == 399:
            print(
                f"Settled {data.time:.2f}s; peak self penetration {peak_self * 1000:.3f} mm",
                flush=True,
            )
    mujoco.mj_forward(model, data)
    save(data, "settled")
    np.savez_compressed(
        ROOT / "outputs/settling_trajectory.npz",
        time=np.array([t[0] for t in trajectory]),
        qpos=np.array([t[1] for t in trajectory]),
        qvel=np.array([t[2] for t in trajectory]),
        ctrl=data.ctrl,
    )
    final_nodes = nodes(model, data)
    length_error = max(
        float(
            np.max(
                abs(
                    np.linalg.norm(np.diff(final_nodes[k], axis=0), axis=1)
                    - rest_lengths[k]
                )
            )
        )
        for k in rest_lengths
    )
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number)
        for i, w in enumerate(data.warning)
        if w.number
    }
    report["settling"] = {
        "simulation_s": data.time,
        "wall_s": time.monotonic() - start,
        "warnings": warnings,
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "peak_sampled_penetration_m": peak,
        "peak_sampled_cable_self_penetration_m": peak_self,
        "penetrating_self_contact_samples": self_contact_samples,
        "maximum_segment_length_error_m": length_error,
        "final_contacts": contact_summary(model, data),
        "final_crossings": crossings(final_nodes),
        "observation": observe(model, data),
    }
    active = []
    for i, c in enumerate(data.contact):
        names = [model.geom(g).name for g in [c.geom1, c.geom2]]
        if all("_segment_" in name for name in names) and c.efc_address >= 0:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1e-8:
                active.append(
                    {
                        "geoms": names,
                        "normal_force_N": float(force[0]),
                        "distance_m": float(c.dist),
                    }
                )
    report["settling"]["active_self_contacts_at_end"] = active
    initial, final = report["initial_crossings"], report["settling"]["final_crossings"]
    if initial and final:
        cost = np.array(
            [
                [
                    sum(abs(a[k][1] - b[k][1]) for k in ["a", "b"])
                    if a["a"][0] == b["a"][0] and a["b"][0] == b["b"][0]
                    else 1e5
                    for b in final
                ]
                for a in initial
            ]
        )
        ii, jj = linear_sum_assignment(cost)
        matched = [(i, j) for i, j in zip(ii, jj) if cost[i, j] <= 6]
        same = sum(
            np.sign(initial[i]["height_difference_m"])
            == np.sign(final[j]["height_difference_m"])
            for i, j in matched
        )
        report["settling"]["projected_order_diagnostic"] = {
            "matched": len(matched),
            "same_height_order": int(same),
            "max_segment_index_distance": 6,
            "is_topological_proof": False,
        }
    report["limitations"] = [
        "No grasp, transport or untangling policy validated",
        "Projected crossings are diagnostics, not a topological proof",
        "Discrete collision checks do not guarantee no tunnelling under future fast actions",
        "Headphone dimensions, material and friction are approximate",
    ]
    report["knot_interactions"] = {
        "initial": knot_interactions(report["initial_crossings"], task),
        "settled": knot_interactions(report["settling"]["final_crossings"], task),
    }
    (ROOT / "outputs/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "robot_match": True,
                "initial_penetrations": len(report["initial_contacts"]),
                "initial_crossings": len(report["initial_crossings"]),
                **{
                    k: v
                    for k, v in report["settling"].items()
                    if k
                    not in [
                        "observation",
                        "final_contacts",
                        "final_crossings",
                        "active_self_contacts_at_end",
                    ]
                },
                "final_crossings": len(report["settling"]["final_crossings"]),
            },
            indent=2,
        ),
        flush=True,
    )
    assert not warnings and report["settling"]["finite"]
    assert length_error < 1e-8
    assert peak_self < 0.0002, "Excessive cable self-penetration during settling"


if __name__ == "__main__":
    main()
