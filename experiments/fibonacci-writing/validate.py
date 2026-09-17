"""Check a writing recording and its controller-reported physical metrics."""

import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
r = json.loads((OUT / "results.json").read_text())
with np.load(OUT / "episode.npz") as episode:
    assert np.isfinite(episode["qpos"]).all()
    assert episode["qpos"].shape[1] == m.nq
    assert np.all(np.diff(episode["times"]) > 0)
    assert len(episode["marks"]) == r["marks"]
    assert int(episode["mark_counts"][-1]) == r["marks"]
assert r["complete"] and r["completed_strokes"] == r["strokes"] == 52
assert r["non_tip_board_contact_samples"] == 0
assert r["min_base_height_m"] > 0.70 and r["min_hand_contact_bodies"] >= 2
assert r["max_grip_displacement_mm"] < 40
limits = np.max(np.abs(m.jnt_actfrcrange[m.actuator_trnid[:, 0]]), axis=1)
assert np.all(np.asarray(r["effective_peak_torque_Nm"]) <= limits + 1e-9)
assert m.neq == 0 and m.nmocap == 0
print(
    "PASS: recorded states, 52 strokes, reported stance/contact/grip and joint-torque checks."
)
print("This checks the saved recording and report; it does not rerun the controller.")
