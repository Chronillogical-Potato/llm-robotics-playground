"""Render recorded dynamics with a fixed full-task camera and honest status."""

import argparse
import json
import imageio_ffmpeg
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from environment import ROOT, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--stills", action="store_true")
    parser.add_argument(
        "--output", help="Separate output basename; source data is never overwritten"
    )
    parser.add_argument("--end-time", type=float, help="Last source time to show")
    parser.add_argument(
        "--omit",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
    )
    parser.add_argument(
        "--end-hold",
        type=float,
        default=0.0,
        help="Freeze the final recorded frame for review",
    )
    parser.add_argument("--elevation", type=float, default=-64.0)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--distance", type=float, default=0.78)
    parser.add_argument("--lookat", type=float, nargs=3, default=[-0.035, 0.015, 0.06])
    args = parser.parse_args()
    report = json.loads((ROOT / "outputs" / (args.name + ".json")).read_text())
    tr = np.load(ROOT / "outputs" / (args.name + "_trajectory.npz"))
    if args.speed <= 0 or args.end_hold < 0:
        parser.error("Speed must be positive and end hold nonnegative")
    edited = bool(args.omit or args.end_time is not None or args.end_hold)
    if edited and (not args.output or args.output == args.name):
        parser.error("An edited rendering requires a separate --output basename")
    output = args.output or args.name
    end = min(
        float(tr["time"][-1]),
        args.end_time if args.end_time is not None else float(tr["time"][-1]),
    )
    if end <= 0:
        parser.error("End time must be positive")
    intervals = sorted(args.omit)
    cursor = 0.0
    kept = []
    for a, b in intervals:
        if not 0 <= cursor <= a < b <= end:
            parser.error(
                "Omitted intervals must be ordered, disjoint and inside the selected recording"
            )
        if a > cursor:
            kept.append((cursor, a))
        cursor = b
    if cursor < end:
        kept.append((cursor, end))
    if not kept:
        parser.error("The edit must retain recorded motion")
    duration = sum(b - a for a, b in kept) / args.speed

    def source_time(playback):
        remaining = min(playback, duration) * args.speed
        for a, b in kept:
            if remaining < b - a:
                return a + remaining
            remaining -= b - a
        return end

    frame_times = np.arange(max(2, int(np.ceil((duration + args.end_hold) * 30)))) / 30
    source_times = np.asarray([source_time(t) for t in frame_times])
    # A cut jumps between recorded states. Never synthesize an intermediate
    # cable or robot pose to make an edited trial look physically continuous.
    frame_indices = np.minimum(
        np.searchsorted(tr["time"], source_times, side="right") - 1, len(tr["time"]) - 1
    )
    frame_indices = np.maximum(frame_indices, 0)
    final_index = max(0, int(np.searchsorted(tr["time"], end, side="right")) - 1)
    edit_manifest = {
        "source": args.name,
        "output": output,
        "source_end_s": end,
        "omitted_source_intervals_s": intervals,
        "kept_source_intervals_s": kept,
        "motion_playback_s": duration,
        "end_hold_s": args.end_hold,
        "playback_speed": args.speed,
        "camera": {
            "elevation": args.elevation,
            "distance": args.distance,
            "lookat": args.lookat,
            "azimuth": args.azimuth,
        },
        "continuous_recording": not bool(intervals),
        "pose_interpolation": False,
        "source_scene_sha256": report["scene_sha256"],
        "scope": "Presentation edit of recorded dynamics; no new robot-control or complete-untangling claim.",
    }
    (ROOT / "outputs" / (output + "_render.json")).write_text(
        json.dumps(edit_manifest, indent=2) + "\n"
    )
    model, data = load(args.name + "_start")
    model.light_castshadow[:2] = False
    model.light_diffuse[:2] *= 0.4
    model.light_specular[:2] *= 0.4
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    opt.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    width, height = 1280, 960
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 26)
        small = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 19)
    except OSError:
        font = ImageFont.load_default(size=26)
        small = ImageFont.load_default(size=19)

    def frame(renderer, index):
        for k in ["qpos", "qvel", "ctrl"]:
            getattr(data, k)[:] = tr[k][index]
        # The dynamics are already recorded. Recompute geometry for display;
        # do not solve a second set of forces just to render the same positions.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera=camera, scene_option=opt)
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rounded_rectangle((18, 18, 1130, 96), radius=8, fill="#10191f")
        draw.text(
            (34, 27),
            "Untangling EarPods · "
            + ("Edited physical recording" if intervals else "Recorded physical trial"),
            font=font,
            fill="#eff2ef",
        )
        draw.text(
            (34, 64),
            "Standard ALOHA 2 · Recorded contact dynamics · Simulator-state planning",
            font=small,
            fill="#bbc9ca",
        )
        t = float(tr["time"][index])
        phase = next(
            (p["name"] for p in reversed(report["phases"]) if t >= p["start"]), "start"
        )
        if edited and t >= end - 0.04:
            phase = "End of two-region demonstration"
        draw.rounded_rectangle((18, 867, 1260, 942), radius=8, fill="#10191f")
        origin = (
            "Initial grasp staged"
            if report.get("staged_initial_grasp")
            else "Actuator-driven approach"
            if report.get("start_state") == "settled"
            else "Recorded continuation"
        )
        draw.text(
            (34, 879),
            f"{'Source ' if intervals else ''}{t:.2f} s  ·  {phase}",
            font=small,
            fill="#eff2ef",
        )
        draw.text(
            (34, 908),
            f"{args.speed:g}× playback  ·  "
            + ("One grasp-retry interval omitted" if intervals else origin),
            font=small,
            fill="#bbc9ca",
        )
        after_cut = next(((a, b) for a, b in intervals if b <= t < b + 1.0), None)
        if after_cut:
            draw.rounded_rectangle((18, 806, 660, 854), radius=8, fill="#263b45")
            draw.text(
                (34, 818),
                f"Grasp retries omitted · {after_cut[1] - after_cut[0]:.2f} s",
                font=small,
                fill="#eff2ef",
            )
        if not edited and report.get("error") and t >= float(tr["time"][-1]) - 0.2:
            draw.rounded_rectangle((18, 788, 1258, 854), radius=8, fill="#692e2b")
            draw.text((34, 799), "Attempt stopped", font=font, fill="#fff2e9")
            draw.text((34, 828), report["error"], font=small, fill="#fff2e9")
        elif (
            not edited
            and report.get("outcome_note")
            and t >= float(tr["time"][-1]) - 0.6
        ):
            draw.rounded_rectangle(
                (18, 800, 1258, 854),
                radius=8,
                fill="#204a3b" if report.get("untangled") else "#4a3a24",
            )
            draw.text((34, 815), report["outcome_note"], font=small, fill="#fff2db")
        return im

    with mujoco.Renderer(model, height, width) as renderer:
        frame(renderer, int(frame_indices[0])).save(
            ROOT / "outputs" / (output + "_before.png")
        )
        frame(renderer, final_index).save(ROOT / "outputs" / (output + "_after.png"))
        if args.stills:
            return
        writer = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "30",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(ROOT / "outputs" / (output + ".mp4")),
            ],
            stdin=subprocess.PIPE,
        )
        try:
            for j in frame_indices:
                writer.stdin.write(np.asarray(frame(renderer, int(j))).tobytes())
        finally:
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("ffmpeg failed")
    print(ROOT / "outputs" / (output + ".mp4"), flush=True)


if __name__ == "__main__":
    main()
