"""JSON-lines high-level pilot interface; 0.1 s of physics per accepted command.
Run with --camera for head and wrist images. Models should receive only this
process's I/O, not Python execution access to the trusted simulation process.
"""

import argparse, base64, io, json, sys
import mujoco
from control import Firmware
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", action="store_true")
    args = p.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    f = Firmware(m, d)
    renderer = mujoco.Renderer(m, 180, 320) if args.camera else None

    def observation():
        out = {
            "time": float(d.time),
            "body_position": d.body("robot").xpos.tolist(),
            "body_quaternion_wxyz": d.body("robot").xquat.tolist(),
            "hands": {
                side: {
                    "position": f.hand_pose(side)[0].tolist(),
                    "rotation": f.hand_pose(side)[1].tolist(),
                }
                for side in f.hands
            },
        }
        if renderer:
            from PIL import Image

            out["images"] = {}
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 0
            for camera in ["head", "left_arm_wrist", "right_arm_wrist"]:
                renderer.update_scene(d, camera=camera, scene_option=opt)
                buf = io.BytesIO()
                Image.fromarray(renderer.render()).save(buf, format="PNG")
                out["images"][camera] = base64.b64encode(buf.getvalue()).decode()
        print(json.dumps(out), flush=True)

    observation()
    try:
        for line in sys.stdin:
            try:
                cmd = json.loads(line)
                if not isinstance(cmd, dict) or set(cmd) - {"drive", "hands"}:
                    raise ValueError("Only drive and hands are accepted")
                # Validate using a copy of command state before committing either component.
                import copy

                old_drive = copy.deepcopy(f.drive)
                old_hands = copy.deepcopy(f.hands)
                try:
                    if "drive" in cmd:
                        f.command_drive(**cmd["drive"])
                    for side, hand in cmd.get("hands", {}).items():
                        f.command_hand(side, **hand)
                except (ValueError, TypeError, KeyError):
                    f.drive = old_drive
                    f.hands = old_hands
                    raise
            except (ValueError, TypeError, KeyError) as e:
                print(json.dumps({"error": str(e)}), flush=True)
                continue
            for _ in range(50):
                f.step()
                mujoco.mj_step(m, d)
            observation()
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
