# Dove drawing

A Kinova arm with a Shadow Hand draws a dove using a free pencil held by finger and palm contact.

[Watch the demo · 4×](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/dove-drawing.mp4)

![Dove drawing](preview.png)

## Run

After the [root setup](../../README.md#start-here), run from this directory:

```sh
python run.py
python validate.py
python render.py
```

Or run `python fetch.py dove-drawing` from the repository root and use just validation and rendering. `python render.py --preview` renders selected frames. Generated files go in `outputs/`.

## How it works

`firmware.py` advances contact dynamics and records tip marks. `controller.py` provides arm/wrist motion, and `writer.py` follows the observed pencil tip while adjusting contact pressure. `scene.xml`, `initial_state.json`, and `strokes.json` are the required inputs.

The pencil starts in a prepared, approximately 30° tilted grasp. It has a free joint and no fixed attachment; pickup is outside the task. The controller follows nine paths traced from the supplied reference. It uses simulator state, not live image interpretation. The model wrote and revised the setup and controller with human feedback. Contact parameters approximate a compliant fingertip patch rather than measured materials.

## Recorded outcome

The selected episode completed nine strokes in 173 simulated seconds, with 10,064 contact marks. Median tracking error was 0.149 mm. Recorded telemetry retained at least two loaded hand contact points per step, with no pencil-shaft/paper contact. See `recording.json` for the historical summary.

The video plays at 4× with a camera zoom and 2× display-only graphite width. Recorded contact locations are unchanged. The portable renderer displays the target stroke paths in its reference inset; the original video uses the supplied raster reference. An optional local `reference_dove.png` restores that inset.

Public-package checks include controller startup and a newly simulated first stroke, telemetry validation, replay previews, and compiled-model equivalence. This is one tuned drawing episode, not a general grasping or visual drawing benchmark.

## Assets

[Menagerie Kinova Gen3 and Shadow Hand](../../assets/README.md) retain their original licenses. The user-supplied reference was identified as a Picasso dove; its signature was omitted during tracing. The original raster is not distributed here. `strokes.json` and the artwork depicted in previews/recordings are excluded from the repository's MIT grant; no additional rights to the artwork are granted. Original controller code: [MIT](../../LICENSE).
