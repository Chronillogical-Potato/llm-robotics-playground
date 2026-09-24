"""Fixed geometry, progress accounting, and reward terms for Baoding PPO."""

import torch


def advance_progress(previous, angles, net, high_water, eligible, max_gain):
    # Signed motion always counts. Ineligible motion cannot be cashed in later.
    delta = torch.atan2(torch.sin(angles - previous), torch.cos(angles - previous))
    net = net + delta
    shared = net.min(dim=-1).values
    gain = (shared - high_water).clamp(0, max_gain) * eligible
    return net, torch.maximum(high_water, shared), gain


def geometry_and_risk(position, velocity, center, support):
    radial = (position[..., :2] - center[:2]).norm(dim=-1)
    pair = position[:, 1] - position[:, 0]
    pair_xy = pair[:, :2].norm(dim=-1)
    pair_distance = pair.norm(dim=-1)
    drift = (position.mean(dim=1)[:, :2] - center[:2]).norm(dim=-1)
    height = (position[..., 2] - center[2]).abs()
    speed = velocity.norm(dim=-1)
    valid = ((radial >= .008) & (radial <= .050)).all(1)
    valid &= (pair_xy >= .025) & (pair_distance >= .036) & (pair_distance <= .080)
    valid &= (drift <= .025) & (height <= .035).all(1) & (speed <= .6).all(1)
    supported = (support > .01).all(1)
    risk = torch.stack((
        ((.012 - radial).clamp_min(0) / .004).amax(1),
        ((radial - .040).clamp_min(0) / .010).amax(1),
        (.030 - pair_xy).clamp_min(0) / .005,
        (.038 - pair_distance).clamp_min(0) / .002,
        (pair_distance - .060).clamp_min(0) / .020,
        (drift - .015).clamp_min(0) / .010,
        torch.stack(((height - .025) / .010, (.273 - position[..., 2]) / .015,
                     (position[..., 2] - .355) / .015), dim=-1).clamp_min(0).amax((1, 2)),
        ((speed - .3).clamp_min(0) / .3).amax(1),
    ), dim=1).clamp_max(4).mean(1)
    return valid, supported, risk


def local_ball_coordinates(ball_position, link_position, link_quaternion):
    relative = ball_position[:, :, None, :] - link_position[:, None, :, :]
    quaternion = link_quaternion[:, None, :, :].expand(-1, ball_position.shape[1], -1, -1)
    vector = -quaternion[..., 1:]
    cross = 2 * torch.cross(vector, relative, dim=-1)
    return relative + quaternion[..., :1] * cross + torch.cross(vector, cross, dim=-1)


def middle_side_risk(local):
    # PP, MP, DP segment axes from the official Sharpa model.
    length = local.new_tensor([.047, .0315, .025])
    x = local[..., 0]
    axial = torch.maximum(-x, x - length).clamp_min(0)
    distance = torch.sqrt(axial.square() + local[..., 1].square() + local[..., 2].square())
    nearby = ((.050 - distance) / .020).clamp(0, 1)
    palmar = torch.stack((local[..., 0, 2], local[..., 1, 1], local[..., 2, 1]), -1)
    behind = ((.004 - palmar) / .024).clamp(0, 1)
    return (nearby * behind).amax(dim=(-1, -2))


def middle_posture_risk(joints, reference):
    deviation = ((joints - reference).abs() - .35).clamp_min(0) / .6
    return deviation.clamp_max(1).square().mean(-1)


def motion_costs(actions, previous_actions, joint_velocity, angular_delta, dt, goal_speed):
    return {
        "action_rate": .02 * (actions - previous_actions).square().mean(-1),
        "joint_speed": .01 * (joint_velocity / 2.).square().clamp_max(4).mean(-1),
        "angular_overspeed": .05 * ((angular_delta.abs() / dt / goal_speed - 1)
                                     .clamp_min(0).square().clamp_max(4)).mean(-1),
    }
