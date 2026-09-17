import argparse, json
import numpy as np, mujoco
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
from control import Firmware
from mission import Mission


def run(duration=240, render=False):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    f = Firmware(m, d)
    pilot = Mission(f)
    peaks = np.zeros(m.nu)
    records = []
    states = []
    gait = []
    contacts = {}
    self_contacts = {}
    robot = m.body("robot").id
    obstacles = {
        m.body(n).id: n for n in ["source_table", "destination_table", "barrier"]
    }
    geom_robot = m.body_rootid[m.geom_bodyid] == robot
    geom_chassis = m.geom_bodyid == robot
    geom_leg = np.zeros(m.ngeom, dtype=bool)
    for gid in range(m.ngeom):
        b = int(m.geom_bodyid[gid])
        while b and b != robot:
            if m.body(b).name.startswith("leg_"):
                geom_leg[gid] = True
                break
            b = int(m.body_parentid[b])
    geom_obstacle = np.isin(m.geom_bodyid, list(obstacles))
    for i in range(int(duration / m.opt.timestep)):
        if i % 50 == 0:
            pilot.step()
        f.step()
        mujoco.mj_step(m, d)
        peaks = np.maximum(peaks, np.abs(d.actuator_force))
        # Audit actual solver contacts at every 2 ms physics step, not just video frames.
        gs = d.contact.geom
        bad = (geom_robot[gs[:, 0]] & geom_obstacle[gs[:, 1]]) | (
            geom_robot[gs[:, 1]] & geom_obstacle[gs[:, 0]]
        )
        for cid in np.flatnonzero(bad):
            c = d.contact[cid]
            g1, g2 = c.geom
            rg, og = (g1, g2) if geom_robot[g1] else (g2, g1)
            b = int(m.geom_bodyid[rg])
            while b and not m.body(b).name:
                b = int(m.body_parentid[b])
            key = m.body(b).name + " / " + obstacles[m.geom_bodyid[og]]
            item = contacts.setdefault(
                key,
                {
                    "first_s": float(d.time),
                    "last_s": float(d.time),
                    "samples": 0,
                    "max_penetration_m": 0.0,
                },
            )
            item["last_s"] = float(d.time)
            item["samples"] += 1
            item["max_penetration_m"] = max(item["max_penetration_m"], float(-c.dist))
        bad_self = (geom_leg[gs[:, 0]] & geom_chassis[gs[:, 1]]) | (
            geom_leg[gs[:, 1]] & geom_chassis[gs[:, 0]]
        )
        for cid in np.flatnonzero(bad_self):
            c = d.contact[cid]
            g1, g2 = c.geom
            lg, cg = (g1, g2) if geom_leg[g1] else (g2, g1)
            b = int(m.geom_bodyid[lg])
            while b and not m.body(b).name:
                b = int(m.body_parentid[b])
            key = m.body(b).name + " / chassis"
            item = self_contacts.setdefault(
                key,
                {
                    "first_s": float(d.time),
                    "last_s": float(d.time),
                    "samples": 0,
                    "max_penetration_m": 0.0,
                },
            )
            item["last_s"] = float(d.time)
            item["samples"] += 1
            item["max_penetration_m"] = max(item["max_penetration_m"], float(-c.dist))
        if i % 20 == 0:
            states.append(np.r_[d.time, d.qpos.copy()])
            gait.append([d.time, getattr(f, "gait_period", 1.5), f.speed, f.turn])
        if i % 500 == 0:
            row = {
                "t": round(float(d.time), 2),
                "stage": pilot.stage,
                "robot": d.body("robot").xpos.tolist(),
                "mug": d.body("target_mug").xpos.tolist(),
                "block": d.body("target_block").xpos.tolist(),
            }
            records.append(row)
            print(row, flush=True)
        if pilot.done:
            break
    np.savez_compressed(
        OUT / "trajectory.npz", states=np.array(states), gait=np.array(gait)
    )
    np.savez(OUT / "action_state.npz", qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl)
    (OUT / "action_telemetry.json").write_text(json.dumps(records, indent=2))
    (OUT / "stages.json").write_text(json.dumps(pilot.history, indent=2))
    result = {
        "leg_chassis_contacts": self_contacts,
        "environment_contacts": contacts,
        "contact_check_hz": 1 / m.opt.timestep,
        "time": d.time,
        "stage": pilot.stage,
        "done": pilot.done,
        "mug": d.body("target_mug").xpos.tolist(),
        "block": d.body("target_block").xpos.tolist(),
        "peaks": dict(zip([m.actuator(i).name for i in range(m.nu)], peaks.tolist())),
    }
    (OUT / "action_results.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=240)
    a = p.parse_args()
    run(a.duration)
