# LLM Robotics Playground

Robotics experiments with LLMs and VLMs—from untangling headphones to drawing with robot arms.

Four MuJoCo environments, their controllers, and the recordings behind the demos. Pick an experiment to run it, inspect the result, or try a different approach.

| Experiment | Preview | Recorded result |
| --- | --- | --- |
| [Headphone untangling](experiments/headphone-untangling) | [![Headphones](experiments/headphone-untangling/preview.png)](experiments/headphone-untangling) | Two tangled regions opened; full-release verification remains incomplete. |
| [Robo spider](experiments/robo-spider) | [![Spider](experiments/robo-spider/preview.png)](experiments/robo-spider) | Six legs, two arms, two objects transferred around an obstacle. |
| [Fibonacci writing](experiments/fibonacci-writing) | [![Fibonacci](experiments/fibonacci-writing/preview.png)](experiments/fibonacci-writing) | A humanoid writes a short Python program on a whiteboard. |
| [Dove drawing](experiments/dove-drawing) | [![Dove](experiments/dove-drawing/preview.png)](experiments/dove-drawing) | An articulated hand draws a dove with a friction-held pencil. |

[Watch or download the selected recordings](https://github.com/dimentary/llm-robotics-playground/releases/tag/v0.1.0).

## Start here

Tested on macOS Apple Silicon with Python 3.14 and MuJoCo 3.12.0. Other platforms have not been verified. Rendering requires graphics support; native interactive viewing on macOS uses `mjpython`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check.py
```

Then follow an experiment's README. Every experiment writes generated files into its ignored `outputs/` folder. Robot assets are included once in `assets/`; no model API key is needed to run the saved controllers.

To inspect a recorded run without waiting for new simulation:

```sh
python fetch.py robo-spider
cd experiments/robo-spider
python validate.py
python render.py --preview
```

`fetch.py` downloads only the selected task's replay bundle, checks its SHA-256, and refuses to overwrite existing recordings. Videos and large trajectories live in Releases; [recordings.json](recordings.json) pins their filenames and checksums.

## What the models did

These experiments were developed with Astra through iterative code generation, simulation inspection, and human feedback. The included controllers use simulator state. Running them repeats the controller's behavior; it does not make new model calls or reproduce the original conversation.

Each experiment documents its starting assumptions, model involvement, observed outcome, and presentation edits. Exact model settings were not preserved. These are exploratory examples, not a standardized benchmark or a general capability ceiling. New models and attempts can be added under the same task.

## Credits and license

Original code is [MIT licensed](LICENSE). Robot models retain their upstream licenses; see [asset credits](assets/README.md). The dove artwork and artwork-derived assets are excluded from the MIT grant; see that experiment's credits.
