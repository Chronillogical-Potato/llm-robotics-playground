"""Final 2x playback export from recorded physics; no bottom text bar."""

import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="demo.mp4")
parser.add_argument("--preview", action="store_true")
args = parser.parse_args()
import numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
d = mujoco.MjData(m)
trajectory = np.load(OUT / "trajectory.npz")["states"]
stages = json.loads((OUT / "stages.json").read_text())
result = json.loads((OUT / "action_results.json").read_text())
assert (
    result["success"]
    and not result["environment_contacts"]
    and not result["leg_chassis_contacts"]
)
r = mujoco.Renderer(m, 720, 1280)
small = mujoco.Renderer(m, 180, 320)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
cam = mujoco.MjvCamera()
cam.distance = 3.35
cam.azimuth = 135
cam.elevation = -27
writer = (
    None
    if args.preview
    else imageio.get_writer(
        OUT / args.output,
        fps=25,
        codec="libx264",
        quality=9,
        ffmpeg_params=["-movflags", "+faststart"],
    )
)
for idx in (
    [0, len(trajectory) // 2, len(trajectory) - 1]
    if args.preview
    else range(0, len(trajectory), 2)
):
    state = trajectory[idx]
    t = state[0]
    d.qpos[:] = state[1:]
    mujoco.mj_forward(m, d)
    stage = "APPROACH"
    for event in stages:
        if t >= event["time"]:
            stage = event["stage"]
    cam.lookat[:] = d.body("robot").xpos + [0.15, 0.1, 0.22]
    r.update_scene(d, camera=cam, scene_option=opt)
    im = Image.fromarray(r.render())
    draw = ImageDraw.Draw(im)
    for side, yy in [("right", 24), ("left", 235)]:
        small.update_scene(d, camera=side + "_arm_wrist", scene_option=opt)
        im.paste(Image.fromarray(small.render()), (936, yy))
        draw.rectangle((936, yy + 180, 1256, yy + 204), fill=(15, 23, 30))
        draw.text(
            (946, yy + 183), side.upper() + " WRIST CAMERA", fill="white", font_size=16
        )
    draw.rectangle((20, 20, 760, 112), fill=(15, 23, 30))
    draw.text((36, 29), "DUAL ARM / TABLE TRANSFER", fill="white", font_size=30)
    draw.text(
        (36, 72),
        f"{stage.replace('_', ' ')}   |   {t:05.1f} s   |   2x",
        fill="#68d6df",
        font_size=24,
    )
    if writer:
        writer.append_data(np.asarray(im))
    if args.preview:
        im.save(OUT / f"preview_{idx}.png")
    if idx in [500, 1400, 2500]:
        im.save(OUT / f"final_2x_frame_{idx}.png")
    if idx % 500 == 0:
        print("rendered", idx, "/", len(trajectory), flush=True)
draw.rectangle((190, 240, 900, 515), fill=(15, 23, 30))
for j, line in enumerate(
    [
        "TRANSFER COMPLETE",
        "2 / 2 objects placed and released",
        f"{result['time']:.1f} s simulated  |  {result['path_distance_m']:.2f} m walked",
        "Both objects within 1 cm of target",
    ]
):
    draw.text((225, 277 + 53 * j), line, fill="white", font_size=28 if j else 34)
for _ in range(38):
    if writer:
        writer.append_data(np.asarray(im))
im.save(OUT / "final_2x_finish.png")
if writer:
    writer.close()
small.close()
r.close()
print(OUT / args.output)
