import numpy as np
import mujoco


def grip_observation(model, data, side, object_name):
    contacts = []
    object_bodies = {object_name}
    if "_cable_" in object_name:
        branch, index = object_name.rsplit("_cable_", 1)
        object_bodies = {
            f"{branch}_cable_{i:03}"
            for i in range(max(0, int(index) - 2), int(index) + 3)
        }
    for i, c in enumerate(data.contact):
        bodies = [model.body(model.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
        if not object_bodies.intersection(bodies):
            continue
        finger = next(
            (
                b
                for b in bodies
                if b in [side + "/left_finger_link", side + "/right_finger_link"]
            ),
            None,
        )
        if finger is None or c.efc_address < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if force[0] > 1e-7:
            contacts.append(
                {
                    "finger": finger,
                    "normal_force_N": float(force[0]),
                    "penetration_m": float(max(0, -c.dist)),
                }
            )
    site = data.site(side + "/gripper")
    body = data.body(object_name)
    return {
        "opposing_contacts": len({c["finger"] for c in contacts}) == 2,
        "contacts": contacts,
        "object_position": body.xpos.tolist(),
        "object_in_gripper": (
            site.xmat.reshape(3, 3).T @ (body.xpos - site.xpos)
        ).tolist(),
        "gripper_position": site.xpos.tolist(),
    }


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)
