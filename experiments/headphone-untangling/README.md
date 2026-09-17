# Headphone untangling

Two ALOHA arms work through two tangles in a headphone cable by lifting the wire and pulling the loops apart.

[Watch the demo · 1×, edited](https://github.com/dimentary/llm-robotics-playground/releases/download/v0.1.0/headphone-untangling.mp4)

![Headphone untangling](preview.png)

## Replay

Complete the [setup](../../README.md#setup), then run from the repository root:

```sh
python fetch.py headphone-untangling
cd experiments/headphone-untangling
python render.py --name two_clusters_current --output demo \
  --end-time 29.259 --omit 4.23 5.96 --end-hold 1 \
  --elevation -38 --azimuth 75 --distance .96 --lookat -.035 .015 .07
```

Add `--stills` for the first and last frames. The download contains the source trajectory, start/end states, run report, and edit settings. Generated files go in `outputs/`.

## Run a controller stage

From this directory:

```sh
python validate.py --seconds .4
python run.py --state settled --name middle --cluster middle --stage open --dt .0005
```

This runs the first opening stage. `run.py` works one stage at a time; use the replay above to watch the full demo. Run `python run.py --help` for the next-stage options, and check each saved report before continuing.

`python verify_release.py --state NAME` checks a saved endpoint for full release. For interactive viewing, use `mjpython viewer.py --state initial` on macOS or `python viewer.py --state initial` elsewhere.

## Setup and result

The wire is made of 282 connected segments that can collide with each other. Each tangled region contains a knot and a loop caught in it. The arms use the stock ALOHA controls and hold the wire through contact. The planner reads cable positions from the simulator; I gave feedback on the layout and untangling strategy.

**Both tangled sections open up.** Some loose overlaps remain at the end, and the cable has not been verified as fully untangled.

The video plays the recorded motion at normal speed. It cuts the interval from 4.230 to 5.960 s, ends at 29.259 s, and holds the last pose for one second. Every pose comes from the simulation. The scene, startup, settling, reach, and replay checks pass.

## Credits

[Menagerie ALOHA 2 model](../../assets/README.md); approximate headphone geometry. Inspired by [Qineng Wang's rope-threading experiment](https://x.com/qineng_wang/status/2099893504658866561). Controller code: [MIT](../../LICENSE).
