# Fibonacci writing

A floating-base Unitree G1 holds a marker through finger contact and writes this program on a whiteboard:

```python
a, b = 0, 1
for _ in range(10):
    a, b = b, a + b
    print(a)
```

[Watch the demo · 4×](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/fibonacci-writing.mp4)

![Fibonacci writing](preview.png)

## Run

After the [root setup](../../README.md#start-here), run from this directory:

```sh
python run.py
python validate.py
python ending.py
python render.py --viewer
```

`ending.py` simulates the separate waist-turn ending. For writing alone, skip it and use `python render.py`. Add `--preview` to render selected frames only. Generated files go in `outputs/`.

To inspect the selected recording instead, run `python fetch.py fibonacci-writing` from the repository root, then use validation and rendering without rerunning the writing or ending.

## How it works

`writer.py` follows the glyph paths in `strokes.py`, using IK and contact-force feedback. `scene.xml` and `initial_grasp.npz` define the physical setup. Every visible ink footprint comes from measured, loaded marker/board contact.

The marker starts already grasped; pickup is outside the task. The pregrasp was prepared with temporary support, removed before writing. The humanoid stands through foot contact, and the marker is a free body. Finger gains and contact parameters were tuned. The model authored and revised this controller with human feedback; it is not live image-based model control. The task brief was to physically write a short Fibonacci program, rather than compute numbers during the motion.

## Recorded outcome

The selected run completed all 52 strokes in 272.21 simulated seconds, depositing 4,803 footprints. Median tip tracking error was 0.72 mm; no non-tip board contacts or extra ink during withdrawal were reported. See `recording.json` for the historical summary.

The video uses 4× playback, a synchronized ink inset, a camera zoom, and a separately simulated 12-second waist turn. The renderer trims the unused top of the board for presentation; the physics scene is unchanged.

Public-package checks include controller startup and a newly simulated first stroke, the recorded-state/contact/torque checks, replay previews, and compiled-model equivalence. These checks do not establish robustness across writing styles or starting grasps.

## Assets

[Menagerie Unitree G1](../../assets/README.md), with its original model license. Original code: [MIT](../../LICENSE).
