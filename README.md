# LLM Robotics Playground

Robotics environments and experiments with LLMs and VLMs—from untangling headphones to drawing with robot arms.

A place to share the setups, controllers, recordings, and practical lessons behind these experiments, and to try the same tasks with different models.

## Initial collection

| Experiment | Task | Packaging status |
| --- | --- | --- |
| Headphone untangling | Two robot arms open two tangled regions of a headphone cable. | In preparation |
| Robo spider | A six-legged, dual-arm robot picks up objects, carries them around an obstacle, and places them on another table. | In preparation |
| Fibonacci writing | A humanoid writes a short Fibonacci program on a whiteboard. | In preparation |
| Dove drawing | A robot arm with an articulated hand draws a dove using a pencil. | In preparation |

**The repository currently contains documentation only.** Runnable experiment packages and selected recordings will be added after their dependencies and reproduction steps have been checked.

## What each experiment will include

- A MuJoCo environment, required robot assets, and a defined starting state.
- The controller used for the demonstration, plus instructions to run and inspect it.
- A selected recording and the data needed to replay or render the recorded result.
- Checks, observed outcomes, and known limitations.
- Notes on the model, tools, observations, prompts or task brief, and human guidance used.

Model-authored controllers, model-guided iteration, and live model control are different ways to conduct an experiment. Each package will explain which was used. Replaying a recording or rerunning a saved controller does not rerun the original model interaction.

These are exploratory simulation experiments. Results apply to the documented setup and run; this collection does not yet define a standardized benchmark.

See the [publication plan](docs/publication-plan.md) and [experiment template](docs/experiment-template.md).

## License

Original material in this repository is provided under the [MIT License](LICENSE), unless a file says otherwise. Third-party robot models, images, and other assets retain their own licenses and attribution; the repository license does not relicense them.
