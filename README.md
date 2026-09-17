# LLM Robotics Playground 🤖

Robotics experiments with LLMs and VLMs, from untangling headphones to drawing with robot arms.

I used Astra to build four robot experiments in MuJoCo. You can watch the demos, replay the recordings, or use the environments to try your own approach.

| Experiment | Preview | Demo |
| --- | --- | --- |
| [Headphone untangling](experiments/headphone-untangling) | [![Headphones](experiments/headphone-untangling/preview.png)](experiments/headphone-untangling) | Two robot arms open two tangled cable regions. |
| [Robo spider](experiments/robo-spider) | [![Spider](experiments/robo-spider/preview.png)](experiments/robo-spider) | A six-legged robot carries and places two objects using two arms. |
| [Fibonacci writing](experiments/fibonacci-writing) | [![Fibonacci](experiments/fibonacci-writing/preview.png)](experiments/fibonacci-writing) | A humanoid writes a Python program on a whiteboard. |
| [Dove drawing](experiments/dove-drawing) | [![Dove](experiments/dove-drawing/preview.png)](experiments/dove-drawing) | An articulated hand draws a dove with a pencil. |

[Watch or download the demos](https://github.com/dimentary/llm-robotics-playground/releases/tag/v0.1.0).

## Setup

Tested on macOS Apple Silicon with Python 3.14 and MuJoCo 3.12.0. Rendering needs graphics support. Other platforms have not been verified.

```sh
git clone https://github.com/dimentary/llm-robotics-playground.git
cd llm-robotics-playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check.py
```

Pick an experiment and follow its README to run it. New results go in its `outputs/` folder, which Git ignores. The experiments share the robot models in `assets/`.

## Replay a demo

Download a recorded run without rerunning the simulation:

```sh
python fetch.py robo-spider
cd experiments/robo-spider
python validate.py
python render.py --preview
```

To try another demo, replace `robo-spider` with its folder name and follow that experiment's replay instructions. Downloads are checked against [recordings.json](recordings.json). If replay files already exist in `outputs/`, move them aside before downloading again.

## How I used Astra

I used GPT-6 Astra in Codex to build the environments and write the robot-control code, with feedback from me along the way. The controllers read positions and contacts from the simulator. The code here runs locally without calling a model API. Each experiment's README explains its setup and what the demo shows.

## License

Original code is [MIT licensed](LICENSE). Robot models retain their [upstream licenses](assets/README.md). The dove artwork and artwork-derived assets are excluded from the MIT grant; see the [drawing credits](experiments/dove-drawing/README.md#credits).
