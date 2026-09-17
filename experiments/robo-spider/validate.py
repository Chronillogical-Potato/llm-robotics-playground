import json, hashlib
import numpy as np, mujoco
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
d = mujoco.MjData(m)
state = np.load(OUT / "action_state.npz")
d.qpos[:] = state["qpos"]
d.qvel[:] = state["qvel"]
d.ctrl[:] = state["ctrl"]
mujoco.mj_forward(m, d)
result = json.loads((OUT / "action_results.json").read_text())
result["objects"] = {}
assert result["done"], "Mission did not complete"
if "leg_chassis_contacts" in result:
    assert not result["leg_chassis_contacts"], "Leg contacted chassis"
if "environment_contacts" in result:
    assert not result["environment_contacts"], "Robot contacted a table or barrier"
robot = m.body("robot").id
table = m.body("destination_table").id
for name, xy in [("target_mug", [-0.82, 2.48]), ("target_block", [-1.38, 2.48])]:
    bid = m.body(name).id
    touch_table = False
    touch_robot = False
    for c in d.contact:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bid in [b1, b2]:
            other = b2 if b1 == bid else b1
            touch_table |= other == table
            touch_robot |= m.body_rootid[other] == robot
    error = float(np.linalg.norm(d.body(name).xpos[:2] - xy))
    up = float(d.body(name).xmat[8])
    result["objects"][name] = {
        "on_destination_table": bool(touch_table),
        "touching_robot": bool(touch_robot),
        "placement_error_m": error,
        "upright_cosine": up,
    }
    assert touch_table and not touch_robot and error < 0.03 and up > 0.99
limits = np.maximum(abs(m.actuator_forcerange[:, 0]), abs(m.actuator_forcerange[:, 1]))
assert np.all(m.actuator_forcelimited)
assert all(result["peaks"][m.actuator(i).name] <= limits[i] + 1e-9 for i in range(m.nu))
states = np.load(OUT / "trajectory.npz")["states"]
adr = m.joint("root").qposadr[0] + 1
result["path_distance_m"] = float(
    np.linalg.norm(np.diff(states[:, adr : adr + 2], axis=0), axis=1).sum()
)
result["success"] = True
result["physics_sha256"] = hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest()
result["mujoco_version"] = mujoco.__version__
(OUT / "action_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "peaks"}, indent=2))
