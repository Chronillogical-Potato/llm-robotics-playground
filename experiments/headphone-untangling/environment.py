"""Loading, resetting and observations for the headphone environment."""

from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
import robot as aloha_checks


def geometry_contacts(model, data):
    """Refresh hypothetical rigid geometry and collision distances only.

    Grasp/path scans do not use constraint forces. Avoid constructing and
    factoring the full contact system for every trial finger opening. Actual
    rollouts still execute mj_step and restore full dynamics after a scan.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)


def load(state="initial"):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    data = aloha_checks.initial_state(model)
    if state != "initial":
        saved = np.load(ROOT / "outputs" / f"{state}.npz")
        if (
            "scene_sha256" in saved
            and str(saved["scene_sha256"])
            != hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Saved state belongs to a different scene. Run validate.py again."
            )
        for key in ["qpos", "qvel", "ctrl"]:
            getattr(data, key)[:] = saved[key]
        data.time = float(saved["time"]) if "time" in saved else 0
        mujoco.mj_forward(model, data)
    return model, data


def save(data, name):
    (ROOT / "outputs").mkdir(exist_ok=True)
    np.savez(
        ROOT / "outputs" / f"{name}.npz",
        qpos=data.qpos,
        qvel=data.qvel,
        ctrl=data.ctrl,
        time=data.time,
        scene_sha256=hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest(),
    )


def nodes(model, data):
    task = json.loads((ROOT / "task.json").read_text())
    return {
        name: np.vstack(
            [
                *[data.body(b).xpos for b in info["body_names"]],
                data.site(f"{name}_end").xpos,
            ]
        )
        for name, info in task["headphones"]["cables"].items()
    }


def crossings(points):
    """XY projected crossings for inspection, not a knot invariant or success.

    Records height order so the initial over/under structure is inspectable.
    Near-neighbour segments are omitted because they share material endpoints.
    """
    segments = [
        (name, i, a, b)
        for name, p in points.items()
        for i, (a, b) in enumerate(zip(p[:-1], p[1:]))
    ]
    out = []
    for k, (name_a, ia, a, b) in enumerate(segments):
        for name_b, ib, c, d in segments[k + 1 :]:
            if name_a == name_b and abs(ia - ib) <= 2:
                continue
            matrix = np.column_stack([(b - a)[:2], -(d - c)[:2]])
            det = np.linalg.det(matrix)
            if abs(det) < 1e-12:
                continue
            u, v = np.linalg.solve(matrix, (c - a)[:2])
            if 0 < u < 1 and 0 < v < 1:
                pa, pb = a + u * (b - a), c + v * (d - c)
                out.append(
                    {
                        "a": [name_a, ia],
                        "b": [name_b, ib],
                        "xy_m": pa[:2].tolist(),
                        "height_difference_m": float(pa[2] - pb[2]),
                    }
                )
    return out


def observe(model, data):
    return {
        "time": data.time,
        "cable_nodes": {k: v.tolist() for k, v in nodes(model, data).items()},
        "parts": {
            n: data.body(n).xpos.tolist()
            for n in ["plug", "splitter", "left_earbud", "right_earbud"]
        },
    }
