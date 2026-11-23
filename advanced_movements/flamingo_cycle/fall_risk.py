"""Shared physical fall limits and a geometric guard for imagined transitions."""
import math

import torch


MIN_HEIGHT = .07
MIN_UP = math.cos(math.radians(55))


def margins(state):
    height = state[..., 2] - MIN_HEIGHT
    up = 1 - 2 * state[..., 4:6].square().sum(-1) - MIN_UP
    return height, up


def has_fallen(state):
    height, up = margins(state)
    return (height < 0) | (up < 0)


def terminal_risk(state, logit):
    height, up = margins(state)
    # Smooth approach to the fall boundary gives actor gradients before failure.
    proximity = torch.maximum(torch.sigmoid(-height / .005), torch.sigmoid(-up / .04))
    risk = torch.maximum(logit.sigmoid(), proximity)
    return torch.where(has_fallen(state), torch.ones_like(risk), risk)
