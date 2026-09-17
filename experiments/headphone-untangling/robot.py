"""Validate the unmodified robot, static fixture reach, and short cable settling.

These are environment checks, not a manipulation policy or a grasp success test.
"""

from pathlib import Path
import hashlib
import json
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "assets"
JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def initial_state(model):
    data = mujoco.MjData(model)
    # The upstream 16-DOF keyframe is robot-only. Resetting the whole expanded
    # model to it would move the new free connector to the origin.
    for side in ["left", "right"]:
        for joint, value in zip(JOINTS, [0, -0.96, 1.16, 0, -0.3, 0]):
            data.qpos[model.joint(side + "/" + joint).qposadr[0]] = value
            data.ctrl[model.actuator(side + "/" + joint).id] = value
        for joint in ["left_finger", "right_finger"]:
            data.qpos[model.joint(side + "/" + joint).qposadr[0]] = 0.028
        data.ctrl[model.actuator(side + "/gripper").id] = 0.028
    mujoco.mj_forward(model, data)
    return data


def robot_check(model):
    provenance = json.loads((ROOT / "manifest.json").read_text())
    for file in provenance["files"]:
        if not file["path"].startswith("aloha/"):
            continue
        actual = hashlib.sha256((ROOT / file["path"]).read_bytes()).hexdigest()
        assert actual == file["sha256"], file["path"]
    upstream = mujoco.MjModel.from_xml_path(str(ROOT / "aloha/aloha.xml"))
    checked = {}
    # Compare compiled data too: catches accidental overriding of MJCF defaults.
    fields = {
        "body": [
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
        ],
        "joint": ["jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_actfrcrange"],
        "actuator": [
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_gear",
        ],
        "geom": [
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_type",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "geom_solref",
            "geom_solimp",
        ],
    }
    # World geoms precede robot geoms when compiled. Match unnamed robot geoms
    # by owning body and local order rather than assuming global geom IDs.
    for kind, names in fields.items():
        count = getattr(
            upstream,
            {"body": "nbody", "joint": "njnt", "actuator": "nu", "geom": "ngeom"}[kind],
        )
        indices = np.arange(count)
        if kind == "geom":
            indices = np.array(
                [
                    model.body_geomadr[upstream.geom_bodyid[g]]
                    + g
                    - upstream.body_geomadr[upstream.geom_bodyid[g]]
                    for g in range(count)
                ]
            )
        for name in names:
            assert np.array_equal(
                getattr(model, name)[indices], getattr(upstream, name)
            ), name
        checked[kind] = count
    for name in ["dof_damping", "dof_armature", "dof_frictionloss"]:
        assert np.array_equal(
            getattr(model, name)[: upstream.nv], getattr(upstream, name)
        ), name
    return {
        "revision": provenance["revision"],
        "verified_files": sum(
            f["path"].startswith("aloha/") for f in provenance["files"]
        ),
        "compiled_fields_match": True,
        "objects_checked": checked,
    }


def contacts(model, data, arm=None):
    out = []
    for contact in data.contact:
        bodies = [
            model.body(model.geom_bodyid[g]).name
            for g in [contact.geom1, contact.geom2]
        ]
        if arm and not any(name.startswith(arm + "/") for name in bodies):
            continue
        if contact.dist < -1e-5:
            out.append(
                {
                    "penetration_m": float(-contact.dist),
                    "bodies": bodies,
                    "geoms": [
                        model.geom(g).name for g in [contact.geom1, contact.geom2]
                    ],
                }
            )
    return sorted(out, key=lambda c: -c["penetration_m"])


def reach(model, side, point, yaw=0):
    data = initial_state(model)
    ids = [model.joint(side + "/" + name).id for name in JOINTS]
    addresses = model.jnt_qposadr[ids]
    lower, upper = model.jnt_range[ids].T
    site = model.site(side + "/gripper").id
    # Upstream gripper +X is the finger approach axis. Face down, keeping the
    # actual geometry and upstream gripper site; no virtual wrist extension.
    desired = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("y", np.pi / 2).as_matrix()
    )

    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        return np.r_[
            data.site_xpos[site] - point,
            0.10 * Rotation.from_matrix(desired @ rotation.T).as_rotvec(),
        ]

    seeds = [
        data.qpos[addresses].copy(),
        [0, -0.5, 0.6, 0, 1.3, 0],
        [0, 0.1, -0.8, 0, 2, 0],
    ]
    best = None
    for seed in seeds:
        result = least_squares(
            residual,
            np.clip(seed, lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper),
            max_nfev=160,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        mujoco.mj_forward(model, data)
        hits = contacts(model, data, side)
        item = {
            "arm": side,
            "position_error_m": float(np.linalg.norm(error[:3])),
            "orientation_error_rad": float(np.linalg.norm(error[3:]) / 0.10),
            "joints": result.x.tolist(),
            "contacts": hits,
            "joint_margin_rad": float(
                np.min(np.minimum(result.x - lower, upper - result.x))
            ),
        }
        score = (
            item["position_error_m"]
            + 0.1 * item["orientation_error_rad"]
            + sum(x["penetration_m"] for x in hits)
        )
        if best is None or score < best[0]:
            best = (score, item)
        if (
            item["position_error_m"] < 0.0005
            and item["orientation_error_rad"] < 0.005
            and not hits
        ):
            break
    return best[1]
