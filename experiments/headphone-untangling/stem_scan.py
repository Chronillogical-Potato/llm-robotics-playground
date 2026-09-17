"""Antipodal stem grasps aligned to the observed rigid earbud's 3D axis."""

import numpy as np
import mujoco
from environment import geometry_contacts
from strategy import solve_pose, fingertip_center, grasp_contacts


def scan_stem(m, d, obj, side="left", axial=0.012):
    saved = d.qpos.copy()
    body = d.body(obj)
    axis = body.xmat.reshape(3, 3)[:, 0].copy()
    # Keep the complete pad strip on the stem, away from the cable attachment
    # and the rounded housing. The stem spans body x=.001 to .023 m.
    target = body.xpos + axis * axial
    offset = fingertip_center(m, d, side)
    candidates = []
    for sign in [1, -1]:
        z = axis * sign
        x = np.array([0.0, 0.0, -1.0])
        x -= z * np.dot(x, z)
        if np.linalg.norm(x) < 0.25:
            x = np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])
            x -= z * np.dot(x, z)
        x /= np.linalg.norm(x)
        rotation = np.column_stack([x, np.cross(z, x), z])
        for raise_by in [0.0006, 0.001, 0.0014, 0.0018, 0.0022]:
            d.qpos[:] = saved
            point = target - rotation @ offset - x * raise_by
            sol = solve_pose(m, d, side, point, rotation, forward=False)
            if sol["position_error_m"] > 0.0005 or sol["rotation_error_rad"] > 0.005:
                continue
            for opening in np.arange(0.016, 0.007, -0.0001):
                for finger in ["left_finger", "right_finger"]:
                    d.qpos[m.joint(side + "/" + finger).qposadr[0]] = opening
                geometry_contacts(m, d)
                good, bad = grasp_contacts(m, d, side, obj)
                # A frozen cable cannot represent the small deflection caused
                # by a closing finger. Admit bounded finger/strand contacts as
                # hypotheses only; execution retains all live penetration and
                # self-contact guards. Rigid obstacles remain forbidden.
                flexible = [
                    g
                    for g in bad
                    if any("_cable_" in b for b in g["bodies"])
                    and any(
                        b in [side + "/left_finger_link", side + "/right_finger_link"]
                        for b in g["bodies"]
                    )
                ]
                rigid = [g for g in bad if g not in flexible]
                stem = [g for g in good if obj + "_stem" in g["geoms"]]
                fingers = {
                    b for g in stem for b in g["bodies"] if b.startswith(side + "/")
                }
                if (
                    len(fingers) == 2
                    and not rigid
                    and max([0] + [-g["distance_m"] for g in flexible]) < 0.004
                ):
                    candidates.append(
                        {
                            "arm": side,
                            "target_body": obj,
                            "grasp_mode": "spatial_stem",
                            "site_point": point.tolist(),
                            "rotation": rotation.tolist(),
                            "finger_position": float(opening),
                            "raise_by_m": raise_by,
                            "max_static_overlap_m": max(
                                [0] + [-g["distance_m"] for g in good]
                            ),
                            "contacts": good,
                            "stem_coordinate_m": axial,
                            "flexible_strand_displacements": flexible,
                            **sol,
                        }
                    )
                    break
                if rigid:
                    break
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    print(obj, "3D stem candidates", len(candidates), flush=True)
    return candidates
