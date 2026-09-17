"""Static hypotheses for stock fingertips pinching a selected cable span."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from environment import geometry_contacts
from strategy import solve_pose, fingertip_center


def scan(m, d, branch, index, side="right", spatial=False, direction=None):
    q = d.qpos.copy()
    target = d.geom(f"{branch}_segment_{index:03}").xpos.copy()
    tangent = d.body(f"{branch}_cable_{index:03}").xmat.reshape(3, 3)[:, 0]
    yaw = np.arctan2(tangent[1], tangent[0])
    offset = fingertip_center(m, d, side)
    allowed = {f"{branch}_cable_{i:03}" for i in range(index - 2, index + 3)}
    candidates = []
    for angle in [0, np.pi]:
        rotation = (
            Rotation.from_euler("z", yaw + angle).as_matrix()
            @ Rotation.from_euler("y", np.pi / 2).as_matrix()
        )
        if spatial:
            z = tangent * np.cos(angle)
            approach = np.array(
                direction if direction is not None else [0.0, 0.0, -1.0], dtype=float
            )
            approach -= z * np.dot(approach, z)
            if np.linalg.norm(approach) < 0.25:
                approach = np.array([-1.0 if side == "right" else 1.0, 0.0, 0.0])
                approach -= z * np.dot(approach, z)
            approach /= np.linalg.norm(approach)
            rotation = np.column_stack([approach, np.cross(z, approach), z])
        for raise_by in [0.0006, 0.0010, 0.0014, 0.0018, 0.0022, 0.0026]:
            d.qpos[:] = q
            point = (
                target
                + (
                    -rotation[:, 0] * raise_by
                    if spatial
                    else np.array([0, 0, raise_by])
                )
                - rotation @ offset
            )
            sol = solve_pose(m, d, side, point, rotation, forward=False)
            if sol["position_error_m"] > 0.0005 or sol["rotation_error_rad"] > 0.005:
                continue
            for opening in np.arange(0.012, 0.007, -0.00005):
                for finger in ["left_finger", "right_finger"]:
                    d.qpos[m.joint(side + "/" + finger).qposadr[0]] = opening
                geometry_contacts(m, d)
                good = []
                bad = []
                for c in d.contact:
                    bodies = [m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
                    if not any(b.startswith(side + "/") for b in bodies):
                        continue
                    if any(b in allowed for b in bodies) and any(
                        b in [side + "/left_finger_link", side + "/right_finger_link"]
                        for b in bodies
                    ):
                        if c.dist < 0.00003:
                            good.append((bodies, float(c.dist)))
                    elif c.dist < -0.00003:
                        bad.append((bodies, float(c.dist)))
                touched = {
                    b
                    for bodies, dist in good
                    for b in bodies
                    if b.startswith(side + "/")
                }
                if len(touched) == 2 and not bad:
                    candidates.append(
                        {
                            "arm": side,
                            "target_body": f"{branch}_cable_{index:03}",
                            "site_point": point.tolist(),
                            "rotation": rotation.tolist(),
                            "finger_position": float(opening),
                            "raise_by_m": raise_by,
                            "contacts": good,
                            **sol,
                        }
                    )
                    break
                if bad:
                    break
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return candidates
