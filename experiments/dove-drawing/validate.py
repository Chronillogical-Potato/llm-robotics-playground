"""Check the physical-grasp recording, telemetry, and reported actuator limits."""

import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
r = json.loads((OUT / "physical_dove_results.json").read_text())
assert (
    r["scene_sha256"] == hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest()
)
assert r["completed"] and not r["attachments"] and r["actuator_limit_compliance"]
assert m.neq == 0 and m.nmocap == 0
assert m.jnt_type[m.body("pencil").jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE
with np.load(OUT / "physical_dove_episode.npz") as episode:
    assert np.isfinite(episode["qpos"]).all()
    assert episode["qpos"].shape[1] == m.nq
    assert np.all(np.diff(episode["times"]) > 0)
    assert len(episode["marks"]) == r["contact_stamps"]
    assert np.all(episode["physics"][:, 5] >= 2), "Lost loaded hand contacts"
    assert np.all(episode["physics"][:, -1] == 0), "Pencil shaft contacted paper"
    assert np.all(episode["marks"][:, 6] > 0.015), "Mark without loaded tip contact"
print(
    "PASS: free pencil, recorded hand contacts, tip-only paper contact, reported actuator limits."
)
print("This checks the saved recording and report; it does not rerun the controller.")
