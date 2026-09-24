# LLM Robotics Playground 🤖

A library of experiments using frontier LLMs/VLMs for robotics tasks, currently focusing on GPT-6 Astra.

The earlier demos include MuJoCo environments and controllers. The Baoding experiment includes an Isaac Lab/PhysX trainer, a policy checkpoint, and a recorded rollout with a kinematic MuJoCo replay. Current tasks cover manipulation, locomotion, writing, drawing, and dexterous in-hand motion.

| Experiment | Preview | Demo |
| --- | --- | --- |
| [Headphone untangling](experiments/headphone-untangling) | [![Headphones](experiments/headphone-untangling/preview.png)](experiments/headphone-untangling) | Two robot arms open two tangled cable regions. |
| [Robo spider](experiments/robo-spider) | [![Spider](experiments/robo-spider/preview.png)](experiments/robo-spider) | A six-legged robot carries and places two objects using two arms. |
| [Fibonacci writing](experiments/fibonacci-writing) | [![Fibonacci](experiments/fibonacci-writing/preview.png)](experiments/fibonacci-writing) | A humanoid writes a Python program on a whiteboard. |
| [Dove drawing](experiments/dove-drawing) | [![Dove](experiments/dove-drawing/preview.png)](experiments/dove-drawing) | An articulated hand draws a dove with a pencil. |
| [Baoding balls](experiments/baoding-balls) | [![Baoding balls](experiments/baoding-balls/preview.png)](experiments/baoding-balls) | A Sharpa hand rotates two balls in simulation. |

[Watch or download the earlier demos](https://github.com/dimentary/llm-robotics-playground/releases/tag/v0.1.0).

## Setup

Tested on macOS Apple Silicon with Python 3.14 and MuJoCo 3.12.0. Rendering needs graphics support. Other platforms have not been verified.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and install its dependencies:

```sh
git clone https://github.com/dimentary/llm-robotics-playground.git
cd llm-robotics-playground
uv sync --locked
uv run python check.py
```

uv handles Python 3.14 and the project environment. Pick an experiment and follow its README to run it. New results go in its `outputs/` folder, which Git ignores. The earlier experiments share robot models in `assets/`; the Baoding model is included with that experiment.

## Replay a demo

Download a recorded run without rerunning the simulation:

```sh
uv run python fetch.py robo-spider
cd experiments/robo-spider
uv run python validate.py
uv run python render.py --preview
```

To try another demo, replace `robo-spider` with its folder name and follow that experiment's replay instructions. Downloads are checked against [recordings.json](recordings.json). If replay files already exist in `outputs/`, move them aside before downloading again.

The Baoding demo is already included with its selected trace and video; it does not use `fetch.py`.

## How the experiments work

The first experiments used GPT-6 Astra in Codex to build the environments and write the robot-control code. I guided the task setup and gave feedback along the way. The Baoding demo instead uses an RL policy trained in Isaac Lab/PhysX. Replay and validation here run locally without calling a model API. Each experiment's README explains its setup and what the demo shows.

## License

Original code is [MIT licensed](LICENSE). Robot models retain their upstream licenses: [shared assets](assets/README.md) and the [Sharpa model](experiments/baoding-balls/model/LICENSE.txt). The dove artwork and artwork-derived assets are excluded from the MIT grant; see the [drawing credits](experiments/dove-drawing/README.md#credits).
