# Baoding balls

A 22-joint Sharpa Wave hand rotates two 19 mm-radius balls in simulation. This is a compact, **evaluation-only** package for one selected PPO baseline: the checkpoint, its recorded PhysX rollout, a validator, and a kinematic replay. It contains no training or tuning pipeline.

[Watch the selected 12-second demo at 1×](demo.mp4)

![Two Baoding balls in the Sharpa hand](preview.png)

## Replay

Complete the [repo setup](../../README.md#setup), then from this directory:

```sh
uv run python validate.py
uv run python replay.py --preview
uv run python replay.py
```

The last command writes `outputs/replay.mp4`. It renders the saved PhysX states at 30 fps with MuJoCo **kinematics only**; it does not re-run the policy or integrate new dynamics. Ball self-orientation is not shown. `demo.mp4` is the already-rendered, verified 1× recording.

## Baseline and result

`baseline.pt` is the original PPO checkpoint (iteration 3320, including optimizer and observation-normalizer state), trained in Isaac Lab/PhysX with 1,024 parallel environments. It is preserved for reference; running the policy requires the original Isaac Lab task, which is intentionally not part of this small replay package. The selected trace is seed 982210, trial 12, from a 12-second, 60 Hz evaluation.

| Selected trial | Recorded result |
| --- | ---: |
| Both balls retained | full first episode |
| Shared net orbital turns | 4.935 |
| Geometry-valid sampled frames | 100% |
| Maximum sampled ball–hand penetration | 0.321 mm |
| All-finger side-risk fraction | 70.65% |

This is a **best-of-archive demonstration**, not the average success rate or a fully validated policy. It was chosen from 592 first-episode records (37 distinct saved evaluation traces) after measured trajectory, joint, and contact checks. The high side-risk score is an unresolved warning, especially around the ring and pinky fingers; it is a geometric heuristic, not a definitive contact classifier. Intermediate physics substeps and the time-zero state were not recorded, so the checks are **not full physics certification**. Reused evaluation seeds and selection from the archive do not establish generalization.

## Provenance and credits

The original full evaluation trace has SHA-256 `e4b9e3fc430ac2bae1f9331b8662cabc94c6e7cf49cf30ac5848835363976866`. `trace.npz` keeps only the selected trial and the fields needed for replay and checks. The unchanged checkpoint SHA-256 is `c777e7411662181d4cb7bcba364dec3d51e4bb87928d51d2eebf743a1ed8a7b4`; the video SHA-256 is `c457bfb915869007ce53440a7b7ce4ce71863f0ed1371b5831ce96819a64c7d0`.

The hand XML and referenced meshes are from [Sharpa's Wave model](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) at commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e`, under [Apache-2.0](model/LICENSE.txt), with its [notice](model/NOTICE.txt). The local replay and validator code are [MIT licensed](../../LICENSE).
