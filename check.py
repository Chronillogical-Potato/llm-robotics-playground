"""Check asset integrity, load each scene, and exercise its controller briefly."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKS = {
    "headphone-untangling": """
from environment import load
import robot, control, run
m, d = load('initial')
robot.robot_check(m)
for _ in range(20): mujoco.mj_step(m, d)
""",
    "robo-spider": """
from control import Firmware
from mission import Mission
m = mujoco.MjModel.from_xml_path('scene.xml'); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
f = Firmware(m, d); pilot = Mission(f); pilot.step()
for _ in range(50): f.step(); mujoco.mj_step(m, d)
""",
    "fibonacci-writing": """
from writer import Writer
from strokes import script_paths
w = Writer(); m, d = w.m, w.d
assert len(script_paths()) == 52
for _ in range(100): w.step()
""",
    "dove-drawing": """
from controller import Controller
from writer import PhysicalFirmware, PhysicalWriter
from run import RecordingPhysical
f = RecordingPhysical(); c = Controller(f); w = PhysicalWriter(c)
c.command({}, .1); m, d = f.model, f.data
assert m.neq == 0 and m.nmocap == 0
""",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=CHECKS)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / "assets" / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Asset mismatch: {path}")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    for name in [args.experiment] if args.experiment else CHECKS:
        code = (
            "import mujoco, numpy as np\n"
            + CHECKS[name]
            + """
assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
assert not any(w.number for w in d.warning), 'MuJoCo warning'
print('PASS: scene, controller startup, short physics step', flush=True)
"""
        )
        print(name, flush=True)
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT / "experiments" / name,
            env=env,
            check=True,
            timeout=120,
        )
    print("These smoke checks do not establish full task success.")


if __name__ == "__main__":
    main()
