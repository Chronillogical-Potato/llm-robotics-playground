"""Inspect the scene in MuJoCo's native viewer (mjpython on macOS).

Starts paused. Space toggles passive physics; R resets to the selected state.
No arm policy or hidden cable animation is applied.
"""

import argparse
import threading
import time
import mujoco
import mujoco.viewer
from environment import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="settled", choices=["initial", "settled"])
    args = parser.parse_args()
    model, data = load(args.state)
    model.light_castshadow[:2] = False
    model.light_diffuse[:2] *= 0.4
    model.light_specular[:2] *= 0.4
    reset_qpos, reset_qvel, reset_ctrl = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
    )
    state = {"running": False, "reset": False}
    lock = threading.Lock()

    def key(code):
        with lock:
            if code == 32:
                state["running"] = not state["running"]
            elif code in [ord("R"), ord("r")]:
                state["reset"] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = [0, 0.02, 0.04]
        viewer.cam.distance = 0.95
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -58
        viewer.opt.geomgroup[3:] = 0
        viewer.opt.sitegroup[:] = 0
        while viewer.is_running():
            start = time.monotonic()
            with lock:
                if state["reset"]:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:], data.qvel[:], data.ctrl[:] = (
                        reset_qpos,
                        reset_qvel,
                        reset_ctrl,
                    )
                    mujoco.mj_forward(model, data)
                    state["reset"], state["running"] = False, False
                running = state["running"]
            if running:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
