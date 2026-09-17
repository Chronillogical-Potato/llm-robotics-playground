from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
from writer import *
from strokes import script_paths, CODE
import json, time

w = Writer()
np.savez(OUT / "initial_state.npz", qpos=w.d.qpos, targets=w.targets)
started = time.monotonic()
w.goto([0.410, -0.015, 1.15], 2)
completed = 0
success = False
try:
    for i, (row, ch, path) in enumerate(script_paths()):
        w.draw(path, 0.018)
        completed += 1
        if i % 10 == 0:
            print(
                i,
                "line",
                row + 1,
                "char",
                ch,
                "t",
                round(w.d.time, 1),
                "marks",
                len(w.marks),
                "slip",
                round(w.maxslip * 1000, 2),
                "wall",
                round(time.monotonic() - started, 1),
                flush=True,
            )
    w.phase = "Finished"
    ink_before_retract = len(w.marks)
    start = w.targets[w.arm].copy()
    mid = start.copy()
    mid[0] += 0.5
    mid[3] = min(2.0, mid[3] + 0.3)
    rest = np.array([0.05, -0.22, 0, 0.5, 0, 0, 0])
    for finish, n in [(mid, 1500), (rest, 2500)]:
        start = w.targets[w.arm].copy()
        for i in range(n):
            t = (i + 1) / n
            s = t * t * t * (10 + t * (-15 + 6 * t))
            w.targets[w.arm] = start + (finish - start) * s
            w.step()
    for _ in range(1000):
        w.step()
    assert len(w.marks) == ink_before_retract, (
        "Marker touched the board during withdrawal"
    )
    success = True
finally:
    w.save()
    a = np.array(w.track)
    error = np.linalg.norm(a[:, 1:3] - a[:, 4:6], axis=1)
    out = dict(
        completed_strokes=completed,
        complete=success,
        max_tip_force_N=w.maxforce,
        min_base_height_m=w.minheight,
        non_tip_board_contact_samples=w.nontip_board_contacts,
        effective_peak_torque_Nm=w.effective_peak.tolist(),
        code=CODE,
        duration_s=w.d.time,
        strokes=len(script_paths()),
        marks=len(w.marks),
        median_tracking_error_mm=float(np.median(error) * 1000),
        p95_tracking_error_mm=float(np.percentile(error, 95) * 1000),
        max_grip_displacement_mm=w.maxslip * 1000,
        min_hand_contact_bodies=w.mincontacts,
        final_base_height_m=float(w.d.qpos[2]),
        peak_actuator_force=w.peak.tolist(),
    )
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(out, flush=True)
