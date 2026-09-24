"""Render the selected PhysX trajectory with MuJoCo kinematics, without simulation."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"


def build_model():
    xml = ET.parse(MODEL / "right_sharpa_wave.xml").getroot()
    xml.find("compiler").set("meshdir", str(MODEL / "meshes"))
    ET.SubElement(xml, "option", timestep=str(1 / 240), gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="80", tolerance="1e-9")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="1440", offheight="1080")
    ET.SubElement(visual, "headlight", diffuse="0.65 0.65 0.65", ambient="0.25 0.25 0.25")
    world = xml.find("worldbody")
    hand = world.find("body")
    hand.set("pos", "0 0 0.25")
    hand.set("quat", "0 0.7071067811865476 0 0.7071067811865476")
    for geom in hand.iter("geom"):
        geom.set("friction", "1.0 0.005 0.0001")
        if geom.get("group") == "1":
            geom.set("rgba", "0.23 0.66 0.40 1" if "elastomer" in geom.get("mesh", "")
                     else "0.85 0.87 0.90 1")
    for i, (position, color) in enumerate((
        ((0.085, -0.020, 0.302), (0.06, 0.48, 0.85)),
        ((0.085, 0.020, 0.302), (0.96, 0.35, 0.10)),
    )):
        ball = ET.SubElement(world, "body", name=f"ball_{i}", pos=" ".join(map(str, position)))
        ET.SubElement(ball, "freejoint", name=f"ball_{i}_joint")
        ET.SubElement(ball, "geom", name=f"ball_{i}_geom", type="sphere", size="0.019",
                      mass="0.035", rgba=" ".join(map(str, (*color, 1))),
                      friction="1.0 0.005 0.0001", solref="0.008 1", condim="4")
    ET.SubElement(world, "geom", type="plane", name="floor", size="2 2 0.1",
                  rgba="0.12 0.15 0.19 1")
    ET.SubElement(world, "light", pos="0.2 -0.4 1.3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.geom_rgba[model.geom("floor").id] = (0.83, 0.81, 0.76, 1)
    return model, mujoco.MjData(model)


def draw_frame(renderer, model, data, joint_addresses, ball_addresses, positions, joints, font, small_font, camera, options):
    data.qpos[joint_addresses] = joints
    for ball, address in enumerate(ball_addresses):
        data.qpos[address:address + 3] = positions[ball]
        data.qpos[address + 3:address + 7] = (1, 0, 0, 0)  # Untextured spheres: spin is not depicted.
    mujoco.mj_forward(model, data)  # Never call mj_step: the trace supplies the motion.
    renderer.update_scene(data, camera=camera, scene_option=options)
    frame = Image.fromarray(renderer.render())
    text = ImageDraw.Draw(frame)
    text.text((24, 22), "Sharpa Baoding", font=font, fill=(30, 35, 40))
    text.text((24, 54), "PhysX replay · 1×", font=small_font, fill=(70, 75, 80))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Render one middle frame instead of the video")
    parser.add_argument("--out", type=Path, help="Output path; defaults inside ignored outputs/")
    args = parser.parse_args()
    out = args.out or HERE / "outputs" / ("preview.png" if args.preview else "replay.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(HERE / "trace.npz") as trace:
        positions = trace["positions"]
        joints = trace["joint_pos"]
        names = trace["joint_names"]
    model, data = build_model()
    joint_addresses = [model.jnt_qposadr[model.joint(str(name)).id] for name in names]
    ball_addresses = [model.jnt_qposadr[model.joint(f"ball_{i}_joint").id] for i in range(2)]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.08, 0, 0.28)
    camera.distance = 0.40
    camera.azimuth = 135
    camera.elevation = -55
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=22)
        small_font = ImageFont.load_default(size=18)

    with mujoco.Renderer(model, height=720, width=960) as renderer:
        def frame_at(t):
            return draw_frame(renderer, model, data, joint_addresses, ball_addresses,
                              positions[t], joints[t], font, small_font, camera, options)

        if args.preview:
            frame_at(2 * (len(positions) // 4)).save(out)
        else:
            with imageio.get_writer(out, fps=30, codec="libx264", quality=8) as writer:
                for t in range(0, len(positions), 2):
                    writer.append_data(np.asarray(frame_at(t)))
    print(f"Saved {out} (kinematic replay of recorded PhysX states)")


if __name__ == "__main__":
    main()
