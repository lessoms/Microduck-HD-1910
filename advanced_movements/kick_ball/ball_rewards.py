"""State-only ball direction cost shared by physics, replay and model imagination."""
import torch


def lateral_ball_penalty(ball_position, initial_position, frame, spec):
    # Use a smooth progress gate: no contact-history flags unavailable to the world model.
    progress = ball_position[..., 0] - initial_position[..., 0]
    gate = ((progress - spec['minimum_forward_m']) / spec['forward_ramp_m']).clamp(0, 1)
    gate = gate * (frame >= spec['start_frame']).to(ball_position.dtype)
    lateral = (ball_position[..., 1] - initial_position[..., 1]).abs()
    excess = (lateral - spec['deadzone_m']).clamp_min(0)
    return spec['weight'] * gate * (1 - torch.exp(-(excess / spec['scale_m']).square()))


def world_loss_weights(dim, schema, position_weight, device):
    if position_weight <= 0:
        raise ValueError('World ball position loss weight must be positive')
    weights = torch.ones(dim, device=device)
    weights[:69] = 2
    a, b = schema['ball']
    weights[a:b] = 3
    weights[a:a+3] = position_weight
    return weights
