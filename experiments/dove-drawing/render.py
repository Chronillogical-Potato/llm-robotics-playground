from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
"""Minimal social-video layout, replaying the existing physical-grip episode."""
import argparse
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from firmware import Firmware, ROOT

SCALE = 6000


def paper_canvas():
    return Image.new(
        "RGB", (round(0.297 * SCALE), round(0.210 * SCALE)), (247, 245, 236)
    )


def stamp(canvas, row, width_scale=1.0):
    p = row[:3]
    force = row[6]
    x = (p[0] - (0.46 - 0.1485)) * SCALE
    y = (0.105 - p[1]) * SCALE
    gray = round(255 * (0.20 + 0.30 * (1 - min(force / 0.4, 1))))
    rad = 0.00030 * SCALE * width_scale
    ImageDraw.Draw(canvas).ellipse(
        (x - rad, y - rad, x + rad, y + rad), fill=(gray, gray, gray)
    )


def scene_marks(scene, rows, width_scale=1.0):
    for row in rows:
        g = scene.geoms[scene.ngeom]
        gray = 0.20 + 0.30 * (1 - min(row[6] / 0.4, 1))
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            np.array([0.00030 * width_scale, 0.00030 * width_scale, 0.000015]),
            row[:3],
            np.eye(3).ravel(),
            np.array([gray, gray, gray, 1.0]),
        )
        scene.ngeom += 1


W, H = 1600, 900
BG = "#080b0d"
WHITE = "#eef0ed"
CYAN = "#98bdb6"
MUTED = "#a7afb2"
BORDER = "#252b2e"
FONT = "/System/Library/Fonts/Avenir Next.ttc"


def font(size, index=0):
    try:
        return ImageFont.truetype(FONT, size, index=index)
    except OSError:
        return ImageFont.load_default(size=size)


TITLE = font(31, index=2)
STATUS = font(22, index=5)
LABEL = font(20, index=5)
PW, PH, FOOTER = 392, 278, 44
STROKE_WIDTH_SCALE = 2.0
from functools import lru_cache


@lru_cache(maxsize=8)
def rounded_mask(size, radius):
    scale = 3
    mask = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, mask.width - 1, mask.height - 1), radius=radius * scale, fill=255
    )
    return mask.resize(size, Image.Resampling.LANCZOS)


def text_at(draw, xy, text, font, fill):
    # Align visible glyphs, rather than font ascent/descent, to the intended position.
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - box[0], xy[1] - box[1]), text, font=font, fill=fill)


def panel_label(draw, x, y, width, text):
    box = draw.textbbox((0, 0), text, font=LABEL)
    text_at(draw, (x + 18, y + (FOOTER - (box[3] - box[1])) / 2), text, LABEL, WHITE)


def inset(frame, picture, xy, label):
    card = Image.new("RGB", (PW, PH + FOOTER), BG)
    card.paste(picture.resize((PW, PH), Image.Resampling.LANCZOS), (0, 0))
    panel_label(ImageDraw.Draw(card), 0, PH, PW, label)
    frame.paste(card, xy, rounded_mask(card.size, 12))


def compose(view, paper, reference, time, phase):
    frame = view.copy()
    if "Settling" in phase:
        stage = "Ready"
    elif "Move" in phase:
        stage = "Approach"
    elif "Lift" in phase:
        stage = "Finished"
    else:
        stage = "Drawing"
    title = "Robot hand / Drawing Picasso"
    title_width = TITLE.getbbox(title)[2]
    header = Image.new("RGB", (title_width + 44, 110), BG)
    d = ImageDraw.Draw(header)
    d.rounded_rectangle(
        (0, 0, header.width - 1, header.height - 1), radius=14, outline=BORDER
    )
    text_at(d, (22, 22), title, TITLE, WHITE)
    text_at(d, (22, 68), stage, STATUS, CYAN)
    sx = 22 + STATUS.getlength(stage) + 19
    text_at(d, (sx, 68), f"·   {time:05.1f} s   ·   4×", STATUS, MUTED)
    frame.paste(header, (32, 32), rounded_mask(header.size, 14))
    # Matched camera-style cards, with a quiet footer and consistent outer margins.
    x = W - 32 - PW
    y = H - 32 - (2 * (PH + FOOTER) + 20)
    inset(frame, paper, (x, y), "Pencil marks")
    inset(frame, reference, (x, y + PH + FOOTER + 20), "Reference")
    return frame


def set_camera(camera, playback_s):
    # Hold the full-arm view for one output second, then ease in over two seconds.
    t = np.clip((playback_s - 1.0) / 2.0, 0.0, 1.0)
    s = t * t * t * (10 + t * (-15 + 6 * t))
    camera.lookat[:] = (1 - s) * np.array([0.30, -0.01, 1.05]) + s * np.array(
        [0.46, -0.01, 0.90]
    )
    camera.distance = (1 - s) * 1.85 + s * 0.84
    camera.azimuth = 40
    camera.elevation = -29 - 2 * s


def target_reference():
    import json

    path = ROOT / "reference_dove.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    canvas = paper_canvas()
    draw = ImageDraw.Draw(canvas)
    for stroke in json.loads((ROOT / "strokes.json").read_text())["strokes"]:
        points = [
            ((x - (0.46 - 0.1485)) * SCALE, (0.105 - y) * SCALE)
            for x, y in stroke["paper_xy_m"]
        ]
        draw.line(points, fill="#303030", width=3)
    return canvas


def main(preview=False):
    a = np.load(OUT / "physical_dove_episode.npz")
    times = a["times"]
    marks = a["marks"]
    f = Firmware("scene.xml", "initial_state.json")
    m, d = f.model, f.data
    r = mujoco.Renderer(m, H, W, max_geom=60000)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    c = mujoco.MjvCamera()
    canvas = paper_canvas()
    prev = 0
    ref = target_reference()
    ref.thumbnail((PW, PH), Image.Resampling.LANCZOS)
    reference = Image.new("RGB", (PW, PH), "white")
    reference.paste(ref, ((PW - ref.width) // 2, (PH - ref.height) // 2))
    selected = np.clip(
        np.searchsorted(times, np.arange(0, times[-1], 4 / 30)), 0, len(times) - 1
    )
    selected = np.r_[selected, np.repeat(len(times) - 1, 60)]
    output_indices = np.arange(len(selected))
    if preview:
        output_indices = np.array(
            [0, 30, 60, 90, len(selected) // 2, len(selected) - 1]
        )
        selected = selected[output_indices]
    writer = (
        None
        if preview
        else imageio.get_writer(
            str(OUT / "physical_dove_simple_rendering.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )
    )
    try:
        for k, i in enumerate(selected):
            d.qpos[:] = a["qpos"][i]
            mujoco.mj_forward(m, d)
            count = int(a["mark_counts"][i])
            set_camera(c, output_indices[k] / 30)
            r.update_scene(d, camera=c, scene_option=opt)
            scene_marks(r.scene, marks[:count], width_scale=STROKE_WIDTH_SCALE)
            view = Image.fromarray(r.render())
            for row in marks[prev:count]:
                stamp(canvas, row, width_scale=STROKE_WIDTH_SCALE)
            prev = count
            frame = compose(view, canvas, reference, times[i], str(a["phases"][i]))
            if writer:
                writer.append_data(np.asarray(frame))
            if preview:
                frame.save(OUT / f"simple_layout_{k}.png")
            elif k in [0, len(selected) // 2, len(selected) - 1]:
                frame.save(OUT / f"simple_review_{k}.png")
            if not preview and k % 300 == 0:
                print("Rendered", k, "/", len(selected), flush=True)
        if not preview:
            frame.save(OUT / "physical_dove_simple.png")
    finally:
        if writer:
            writer.close()
        r.close()
    if not preview:
        (OUT / "physical_dove_simple_rendering.mp4").replace(
            OUT / "physical_dove_simple.mp4"
        )
    print("Preview saved." if preview else str(OUT / "physical_dove_simple.mp4"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    main(p.parse_args().preview)
