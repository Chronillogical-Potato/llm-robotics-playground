# Fibonacci writing

A Unitree G1 humanoid holds a marker and writes a short Fibonacci program on a whiteboard.

[Watch the demo · 4×](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/fibonacci-writing.mp4)

![Fibonacci writing](preview.png)

## Run

Complete the [setup](../../README.md#setup), then run from the repository root:

```sh
cd experiments/fibonacci-writing
python run.py
python validate.py
python ending.py
python render.py --viewer
```

To replay the recorded demo, run `python fetch.py fibonacci-writing` from the repository root, then run only `python validate.py` and `python render.py --viewer` inside this directory. Add `--preview` for selected still frames. All generated files go in `outputs/`.

## Setup and controller

The robot stands on its own feet and starts with the marker already in its hand. The marker is held by finger pressure and friction, with no fixed attachment. The finger controls and contact settings are tuned for writing.

`writer.py` follows the letter paths in `strokes.py` and adjusts the arm using joint positions and contact forces. Ink appears where the marker presses against the board. `ending.py` makes the robot turn toward the camera after writing.

## Result

The robot finishes **52 strokes in 272.21 simulated seconds**. The marker leaves **4,803 contact marks**, with a median distance of **0.72 mm** from the intended path. Only the marker tip touches the board, and it leaves no extra ink as the arm pulls away. The downloaded `outputs/results.json` contains the measurements.

The video plays at 4× with a close-up of the writing. It zooms in as the robot works and includes the final turn toward the camera. The unused top of the board is cropped in the render. Startup, a fresh first stroke, the saved results, and replay rendering have been checked.

## Credits

[Menagerie Unitree G1 model](../../assets/README.md). Controller code: [MIT](../../LICENSE).
