from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
"""Full dove attempt with a free pencil supported only by physical hand contact."""
import json, hashlib
import numpy as np
import mujoco
from firmware import ROOT
from writer import PhysicalFirmware, PhysicalWriter, relative
from controller import Controller


class RecordingPhysical(PhysicalFirmware):
    def __init__(self):
        super().__init__()
        self.state_path = ROOT / "initial_state.json"
        self.frames = []
        self.frame_time = -1
        self.samples = []
        self.phase = "Physical pre-grasp"

    def step(self):
        force = super().step()
        m, d = self.model, self.data
        p = m.body("pencil").id
        hand_contacts = 0
        normal = 0.0
        non_tip_paper = 0
        for ci, ct in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[ct.geom1], m.geom_bodyid[ct.geom2]
            if p in [b1, b2]:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                other = b2 if b1 == p else b1
                if m.body(other).name.startswith("rh_") and f[0] > 0.001:
                    hand_contacts += 1
                    normal += f[0]
                if (
                    self.paper_geom in [ct.geom1, ct.geom2]
                    and self.tip_geom not in [ct.geom1, ct.geom2]
                    and f[0] > 0.001
                ):
                    non_tip_paper += 1
        pos, axis = relative(self)
        self.samples.append(
            [
                d.time,
                force,
                *d.site_xpos[self.tip_site],
                hand_contacts,
                normal,
                *pos,
                *axis,
                non_tip_paper,
            ]
        )
        if d.time - self.frame_time >= 1 / 30 - 1e-6:
            self.frames.append((d.time, d.qpos.copy(), len(self.marks), self.phase))
            self.frame_time = d.time
        return force


if __name__ == "__main__":
    f = RecordingPhysical()
    c = Controller(f)
    w = PhysicalWriter(c)
    f.phase = "Settling physical grasp"
    c.command({}, 1)
    f.phase = "Move to paper"
    w.goto([0.43, 0, 0.794], 2.5)
    reference = json.loads((ROOT / "strokes.json").read_text())["strokes"]
    try:
        for i, stroke in enumerate(reference):
            f.phase = f"Physical stroke {i + 1} / 9"
            before = len(f.marks)
            w.draw(stroke["paper_xy_m"], 0.008)
            print(
                f.phase,
                "time",
                round(f.data.time, 2),
                "marks",
                len(f.marks) - before,
                flush=True,
            )
        f.phase = "Lift after drawing"
        w.goto([0.61, -0.13, 0.90], 2)
        c.command({}, 1)
        completed = True
    except Exception as ex:
        print("episode stopped:", ex, flush=True)
        f.phase = "Controller stopped"
        completed = False
    a = np.array(w.track)
    physics = np.array(f.samples)
    e = np.linalg.norm(a[:, :2] - a[:, 2:4], axis=1) * 1000
    loaded = a[:, 5][a[:, 5] > 0.015]
    np.savez_compressed(
        OUT / "physical_dove_episode.npz",
        times=np.array([r[0] for r in f.frames]),
        qpos=np.array([r[1] for r in f.frames]),
        mark_counts=np.array([r[2] for r in f.frames]),
        phases=np.array([r[3] for r in f.frames]),
        marks=np.array([[*p, *q, v] for p, q, v in f.marks]),
        physics=physics,
        track=a,
    )
    (OUT / "physical_dove_commands.json").write_text(json.dumps(c.actions))
    result = {
        "completed": completed,
        "mode": "GPT-authored scripted controller, free-pencil physical pre-grasp; no live model piloting",
        "attachments": False,
        "free_pencil_dofs": 6,
        "equality_constraints": f.model.neq,
        "duration_sim_s": f.data.time,
        "contact_stamps": len(f.marks),
        "tracking_error_mm": {
            "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)),
            "max": float(e.max()),
        },
        "tip_force_N": {
            "median_loaded": float(np.median(loaded)) if len(loaded) else 0,
            "p95_loaded": float(np.percentile(loaded, 95)) if len(loaded) else 0,
            "max": float(f.max_force),
        },
        "hand_contact_fraction": float(np.mean(physics[:, 5] >= 2)),
        "min_hand_contacts": int(physics[:, 5].min()),
        "shaft_paper_contact_steps": int(physics[:, -1].sum()),
        "actuator_limit_compliance": bool(
            np.all(
                f.peak_torque
                <= np.max(np.abs(f.model.actuator_forcerange), axis=1) + 1e-6
            )
        ),
        "scene_sha256": hashlib.sha256((ROOT / "scene.xml").read_bytes()).hexdigest(),
    }
    (OUT / "physical_dove_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
