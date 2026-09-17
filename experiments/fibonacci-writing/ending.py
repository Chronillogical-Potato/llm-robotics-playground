from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
"""A physical waist turn toward the camera, appended after the writing episode."""
from writer import *
import json


def main():
    a = np.load(OUT / "episode.npz")
    w = Writer()
    w.d.qpos[:] = a["qpos"][-1]
    dt = float(a["times"][-1] - a["times"][-2])
    mujoco.mj_differentiatePos(w.m, w.d.qvel, dt, a["qpos"][-2], a["qpos"][-1])
    w.d.time = float(a["times"][-1])
    w.targets[w.arm] = [0.05, -0.22, 0, 0.5, 0, 0, 0]
    mujoco.mj_forward(w.m, w.d)
    w.initialrel = w.relative()[0]
    w.phase = "Finished"
    start = w.targets[12]
    target = np.deg2rad(-145)
    for i in range(12000):
        t = min((i + 1) / 10000, 1)
        s = t * t * t * (10 + t * (-15 + 6 * t))
        w.targets[12] = start + (target - start) * s
        w.step()
        if i % 2000 == 0:
            print(
                "ending",
                round(w.d.time, 2),
                "waist",
                round(np.rad2deg(w.d.qpos[w.qa[12]]), 1),
                "base",
                round(w.d.qpos[2], 4),
                "slip mm",
                round(w.maxslip * 1000, 2),
                flush=True,
            )
    assert not w.marks, "Ending deposited ink"
    assert w.nontip_board_contacts == 0, "Ending made non-tip board contact"
    limits = np.max(np.abs(w.m.jnt_actfrcrange[w.jids]), axis=1)
    assert np.all(w.effective_peak <= limits + 1e-9)
    np.savez_compressed(
        OUT / "ending_episode.npz",
        times=np.array([f[0] for f in w.frames]),
        qpos=np.array([f[1] for f in w.frames]),
        mark_counts=np.full(len(w.frames), int(a["mark_counts"][-1])),
        phases=np.full(len(w.frames), "Finished"),
    )
    result = dict(
        waist_turn_degrees=float(np.rad2deg(w.d.qpos[w.qa[12]])),
        duration_s=12,
        minimum_base_height_m=w.minheight,
        maximum_grip_displacement_mm=w.maxslip * 1000,
        ink_added=len(w.marks),
        non_tip_board_contact_samples=w.nontip_board_contacts,
        effective_joint_torque_limits_passed=True,
        head_joint_added=False,
    )
    (OUT / "ending_results.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
