"""Check the selected recorded rollout; this is not a new physics evaluation."""

from pathlib import Path

import numpy as np


TRACE = Path(__file__).with_name("trace.npz")


def main():
    with np.load(TRACE) as trace:
        position = trace["positions"]
        joints = trace["joint_pos"]
        ending = trace["endings"]
        support = trace["support_n"]
        risk = trace["finger_dorsal_risk"]
        penetration = trace["ball_hand_penetration_m"]
        violation = trace["joint_violation_rad"]
        center = trace["center"]
        dt = float(trace["dt"])

        assert position.shape == (719, 2, 3)
        assert joints.shape == (719, 22)
        assert trace["joint_names"].shape == (22,)
        assert ending.shape == (719, 2)  # dropped, timed out
        assert support.shape == (719, 2)
        assert risk.shape == (719, 4)
        assert penetration.shape == (719, 2)
        assert violation.shape == (719,)
        assert np.isclose(dt, 1 / 60)
        assert all(np.isfinite(x).all() for x in (position, joints, support, risk, penetration, violation))
        assert not ending[:, 0].any() and ending[-1, 1] and ending[:-1, 1].sum() == 0

        radial = position[:, :, :2] - center[:2]
        pair = position[:, 0, :2] - position[:, 1, :2]
        theta = np.column_stack((
            np.arctan2(radial[:, 0, 1], radial[:, 0, 0]),
            np.arctan2(radial[:, 1, 1], radial[:, 1, 0]),
            np.arctan2(pair[:, 1], pair[:, 0]),
        ))
        delta = np.diff(theta, axis=0)
        turns = np.arctan2(np.sin(delta), np.cos(delta)).sum(axis=0) / (2 * np.pi)
        shared = float(turns.min())
        side_fraction = float((risk.max(axis=1) > 0.25).mean())
        max_penetration_mm = float(penetration.max() * 1000)
        supported_fraction = (support > 0.01).mean(axis=0)

        assert np.isclose(shared, 4.9347968101501465, atol=1e-4)
        assert (supported_fraction > 0.99).all()
        assert max_penetration_mm < 2.0
        assert violation.max() < 0.02
        assert np.isclose(side_fraction, 0.7065368567454798, atol=1e-5)

    print(f"Recorded first episode: {(len(position) - 1) * dt:.2f}s, both balls retained")
    print(f"Shared net turns: {shared:.3f}")
    print(f"Supported frames: {supported_fraction.min():.2%} minimum across balls")
    print(f"Maximum sampled penetration: {max_penetration_mm:.3f} mm")
    print(f"All-finger side-risk fraction: {side_fraction:.2%} (warning, not a pass)")
    print("Recorded-frame checks only; no physics substep or generalization claim.")


if __name__ == "__main__":
    main()
