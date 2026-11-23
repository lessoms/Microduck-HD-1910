"""Stateless landing blend/rate limit shared by physics, replay and imagination."""
import torch


def landing_action(main, lower, previous, frame, start, config):
    if config is None:
        return torch.where((frame >= start)[:, None], lower, main)
    start = config.get('start_frame', start)
    ticks = torch.as_tensor(config['blend_ticks'], device=main.device, dtype=main.dtype)
    limit = torch.as_tensor(config['max_delta_rad'], device=main.device, dtype=main.dtype)
    phase = ((frame-start).to(main.dtype)/ticks).clamp(0, 1)
    weight = phase.square()*(3-2*phase)
    desired = torch.lerp(main, lower, weight[:, None])
    if limit.ndim: limit = limit[:, None]
    smooth = previous + (desired-previous).clamp(-limit, limit)
    return torch.where((frame >= start)[:, None], smooth, main)
