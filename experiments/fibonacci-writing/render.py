"""Render recorded dynamics; ink is deposited only at measured board contacts."""

from pathlib import Path
import argparse
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
W, H = 1920, 1080
FONT = "/System/Library/Fonts/Avenir Next.ttc"


def font(size, index=0):
    try:
        return ImageFont.truetype(FONT, size, index=index)
    except OSError:
        return ImageFont.load_default(size=size)


TITLE = font(30, index=2)
LABEL = font(22, index=5)
INSET_W, INSET_H, INSET_FOOTER = 704, 392, 44
INSET_LABEL = font(24, index=5)


def marks_scene(scene, marks):
    # Marker footprints lie on the vertical surface, normal along world X.
    for row in marks:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            np.array([0.00008, 0.0011, 0.0011]),
            row[:3],
            np.eye(3).ravel(),
            np.array([0.045, 0.085, 0.13, 1]),
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def board_image(marks):
    im = Image.new("RGB", (1800, 1000), (245, 247, 242))
    d = ImageDraw.Draw(im)
    for row in marks:
        x = (0.04 - row[1]) * 3600
        y = (1.18 - row[2]) * 3600
        r = 4
        d.ellipse((x - r, y - r, x + r, y + r), fill=(16, 26, 36))
    return im


def add_board_inset(frame, marks):
    picture = board_image(marks).resize((INSET_W, INSET_H), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (INSET_W, INSET_H + INSET_FOOTER), "#090d11")
    card.paste(picture, (0, 0))
    draw = ImageDraw.Draw(card)
    label = "Written code"
    box = draw.textbbox((0, 0), label, font=INSET_LABEL)
    draw.text(
        (20 - box[0], INSET_H + (INSET_FOOTER - (box[3] - box[1])) / 2 - box[1]),
        label,
        font=INSET_LABEL,
        fill="#f0f1ed",
    )
    mask = Image.new("L", (INSET_W * 3, card.height * 3), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, mask.width - 1, mask.height - 1), radius=53, fill=255
    )
    mask = mask.resize(card.size, Image.Resampling.LANCZOS)
    frame.paste(card, (W - 32 - INSET_W, 32), mask)
    return frame


def header(frame, t, phase):
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle(
        (32, 32, 520, 132), radius=16, fill="#090d11", outline="#28323a", width=1
    )
    d.text((52, 47), "Unitree G1 / Writing Python", font=TITLE, fill="#f0f1ed")
    label = (
        "Writing"
        if phase == "Writing"
        else ("Finished" if phase == "Finished" else "Positioning")
    )
    d.text((52, 91), f"{label}  ·  {t:05.1f} s  ·  4×", font=LABEL, fill="#a4c4bb")
    return frame


def set_lighting(model):
    # Blend the original lighting 75% toward the softer, warmer preview.
    arrays = [
        model.vis.headlight.ambient,
        model.vis.headlight.diffuse,
        model.vis.headlight.specular,
        model.light_diffuse,
        model.light_specular,
        model.mat_rgba,
        model.mat_specular,
        model.mat_shininess,
        model.mat_reflectance,
    ]
    originals = [a.copy() for a in arrays]
    # A warm key with restrained cool fill; avoid additive lights clipping the shell.
    model.vis.headlight.ambient[:] = [0.24, 0.24, 0.24]
    model.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
    model.vis.headlight.specular[:] = [0.02, 0.02, 0.02]
    model.light_diffuse[0] = [0.32, 0.33, 0.35]
    model.light_specular[0] = [0.015, 0.015, 0.02]
    model.light_castshadow[0] = False
    model.light_diffuse[1] = [0.76, 0.71, 0.65]
    model.light_specular[1] = [0.12, 0.105, 0.085]
    model.light_diffuse[2] = [0.28, 0.31, 0.35]
    model.light_specular[2] = [0.015, 0.02, 0.025]
    metal = model.material("metal").id
    model.mat_rgba[metal] = [0.67, 0.68, 0.69, 1]
    model.mat_specular[metal] = 0.12
    model.mat_shininess[metal] = 0.25
    black = model.material("black").id
    model.mat_rgba[black] = [0.12, 0.135, 0.15, 1]
    model.mat_specular[black] = 0.14
    model.mat_shininess[black] = 0.3
    floor = model.material("floor_mat").id
    model.mat_rgba[floor] = [0.16, 0.18, 0.20, 1]
    model.mat_reflectance[floor] = 0.025
    for array, original in zip(arrays, originals):
        array[:] = 0.75 * array + 0.25 * original


def set_board_dimensions(model):
    # Remove 20% of the outer board height from the unused top, preserving the
    # writing surface plane, bottom edge, and frame thickness during replay.
    trim = 2 * model.geom_size[model.geom("board_frame").id, 2] * 0.2
    for name in ("board_frame", "whiteboard"):
        gid = model.geom(name).id
        model.geom_size[gid, 2] -= trim / 2
        model.geom_pos[gid, 2] -= trim / 2


def set_camera(camera, playback_s, height_offset=0.3, close_distance=0.976, azimuth=20):
    # Raise the camera in world space while keeping its horizontal position and focus.
    t = float(np.clip((playback_s - 1) / 2, 0, 1))
    s = t * t * t * (10 + t * (-15 + 6 * t))
    camera.lookat[:] = (1 - s) * np.array([0.18, -0.14, 0.82]) + s * np.array(
        [0.27, -0.17, 1.08]
    )
    original_distance = (1 - s) * 2.5 + s * close_distance
    horizontal = original_distance * np.cos(np.deg2rad(10))
    vertical = original_distance * np.sin(np.deg2rad(10)) + height_offset
    camera.distance = float(np.hypot(horizontal, vertical))
    camera.elevation = -float(np.rad2deg(np.arctan2(vertical, horizontal)))
    camera.azimuth = azimuth


def main(preview=False, viewer=False):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    set_lighting(m)
    set_board_dimensions(m)
    d = mujoco.MjData(m)
    base = np.load(OUT / "episode.npz")
    a = {key: base[key] for key in base.files}
    if viewer:
        end = np.load(OUT / "ending_episode.npz")
        for key in ("times", "qpos", "mark_counts", "phases"):
            a[key] = np.concatenate([a[key], end[key]])
    times = a["times"]
    marks = a["marks"]
    r = mujoco.Renderer(m, H, W, max_geom=50000)
    v = mujoco.MjvOption()
    v.geomgroup[3] = 0
    v.sitegroup[:] = 0
    c = mujoco.MjvCamera()
    selected = np.searchsorted(times, np.arange(0, times[-1], 4 / 30))
    selected = np.minimum(selected, len(times) - 1)
    selected = np.r_[selected, np.repeat(len(times) - 1, 60 if viewer else 75)]
    ids = np.arange(len(selected))
    if preview:
        ids = np.array([0, 60, 375, 600, 825, len(ids) - 1])
    output = "demo.mp4" if viewer else "writing.mp4"
    writer = (
        None
        if preview
        else imageio.get_writer(
            str(OUT / "whiteboard_rendering.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )
    )
    try:
        for ki, oi in enumerate(ids):
            i = selected[oi]
            d.qpos[:] = a["qpos"][i]
            mujoco.mj_forward(m, d)
            set_camera(c, oi / 30)
            r.update_scene(d, camera=c, scene_option=v)
            count = int(a["mark_counts"][i])
            marks_scene(r.scene, marks[:count])
            frame = header(Image.fromarray(r.render()), times[i], str(a["phases"][i]))
            add_board_inset(frame, marks[:count])
            if writer:
                writer.append_data(np.array(frame))
            if preview:
                frame.save(OUT / f"warmer_balanced_preview_{ki}.png")
            if ki % 300 == 0:
                print("render", ki, "/", len(ids), flush=True)
        frame.save(OUT / "whiteboard_result.png")
        board_image(marks).save(OUT / "written_code.png")
    finally:
        if writer:
            writer.close()
        r.close()
    if not preview:
        (OUT / "whiteboard_rendering.mp4").replace(OUT / output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    main(args.preview, args.viewer)
