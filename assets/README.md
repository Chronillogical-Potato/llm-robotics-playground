# Robot assets

Selected, unmodified files from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad), pinned to commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`.

| Model | Used by | License |
| --- | --- | --- |
| ALOHA 2 | Headphone untangling | [BSD-3-Clause](aloha/LICENSE), Trossen Robotics |
| Kinova Gen3 | Robo spider, dove drawing | [BSD-3-Clause](kinova_gen3/LICENSE), Kinova |
| Robotiq 2F-85 | Robo spider | [BSD-2-Clause](robotiq_2f85/LICENSE), ROS-Industrial |
| Unitree G1 | Fibonacci writing | [BSD-3-Clause](unitree_g1/LICENSE), Unitree Robotics |
| Shadow Hand | Dove drawing | [Apache-2.0](shadow_hand/LICENSE) |

`manifest.json` records hashes of the distributed upstream files. `check.py` verifies them. Task scenes use these meshes and are stored separately in each experiment; their controllers and task-specific changes are described there.
