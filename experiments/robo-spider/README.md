# Robo spider

A six-legged robot with two Kinova arms picks up a mug and block, carries them around a barrier, and releases them on another table.

[Watch the demo · 2×](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/robo-spider.mp4)

![Robo spider](preview.png)

## Run

Complete the [setup](../../README.md#setup), then run from the repository root:

```sh
cd experiments/robo-spider
uv run python run.py
uv run python validate.py
uv run python render.py
```

To replay the recorded demo, run `uv run python fetch.py robo-spider` from the repository root, then run the validation and rendering commands above. Add `--preview` to the render command for three still frames. All generated files go in `outputs/`.

## Setup and controller

The robot holds the mug and block through grip pressure and friction. `mission.py` tells it where to go and when to pick up or release each object. `control.py` handles the legs, arms, and grippers. It reads positions from the simulator.

For new model experiments, `pilot.py` provides a JSON-lines interface for driving and moving the hands. Its `--camera` option adds head and wrist images.

## Result

Both objects end upright on the second table after **97.6 simulated seconds**, within **1.9 mm and 2.5 mm** of their targets. The robot stayed clear of the tables and barrier, and its legs stayed clear of its body. Motor forces stayed within the limits set in the simulation.

A fresh run reproduced the saved trajectory exactly. The video plays at 2× and ends with a result card. Detailed metrics are in the downloaded `outputs/action_results.json`.

## Credits

Original six-legged body with [Menagerie Kinova Gen3 and Robotiq 2F-85 models](../../assets/README.md). Controller code: [MIT](../../LICENSE).
