# Robo spider

A six-legged robot with two Kinova arms picks up a mug and block, carries them around a barrier, and releases them on another table.

[Watch the demo · 2×](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/robo-spider.mp4)

![Robo spider](preview.png)

## Run

After the [root setup](../../README.md#start-here), run from this directory:

```sh
python run.py
python validate.py
python render.py
```

Or download the selected recording from the repository root with `python fetch.py robo-spider`, then run just validation and rendering. `python render.py --preview` produces three frames. Generated files go in `outputs/`.

## How it works

`mission.py` sets the route and manipulation stages. `control.py` handles gait, arm IK, bounded actuator commands, and gripper targets. `scene.xml` is the final assembled robot and environment. `pilot.py` also exposes a JSON-lines drive/hand interface; `--camera` adds head and wrist images.

The model wrote and revised the environment and controller with human feedback. The recorded controller uses known world coordinates and simulator state; its available camera interface was not used as a live VLM policy. The task brief was to make an unusual legged, dual-arm robot perform a useful transfer.

## Recorded outcome

Both objects were placed upright and released after 97.602 simulated seconds. Placement errors were 1.9 mm and 2.5 mm. The historical run checked contacts every physics step (500 Hz), with no robot/table/barrier or leg/chassis contacts, and respected configured actuator limits. Small historical metrics and scene hashes are in `recording.json`.

The objects remain free bodies, held by contact and friction. This is a tuned route through one fixed layout. General navigation, perturbations, and physical hardware were not evaluated.

The selected video plays recorded states at 2× and adds a final result card. Public-package checks include a full controller rerun matching the recorded trajectory exactly, saved-run validation, replay previews, and compiled-model equivalence after asset-path changes.

## Assets

Original six-legged body, with [Menagerie Kinova and Robotiq assets](../../assets/README.md). These retain their own licenses. Original code: [MIT](../../LICENSE).
