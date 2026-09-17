# Robot assets

These robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad), pinned to commit `8161bba264d7fa7c99ca301e91e7fb44737676ad` and shared across experiments.

| Model | Used by | License |
| --- | --- | --- |
| ALOHA 2 | Headphone untangling | [BSD-3-Clause](aloha/LICENSE), Trossen Robotics |
| Kinova Gen3 | Robo spider, dove drawing | [BSD-3-Clause](kinova_gen3/LICENSE), Kinova |
| Robotiq 2F-85 | Robo spider | [BSD-2-Clause](robotiq_2f85/LICENSE), ROS-Industrial |
| Unitree G1 | Fibonacci writing | [BSD-3-Clause](unitree_g1/LICENSE), Unitree Robotics |
| Shadow Hand | Dove drawing | [Apache-2.0](shadow_hand/LICENSE) |

The model files are unchanged from that source. Run `uv run python check.py` from the repository root to compare them with the checksums in [manifest.json](manifest.json). Each experiment has its own `scene.xml` for the task setup.
