"""Build future reference features and run the main and landing policies."""
import json

import numpy as np
import torch

from dynamics import JOINT_NAMES
from environment_variations import bounded_action
from landing_control import landing_action
from microduck_sm.math_fk import normalize, qconj, qmul, quat6, rotate


def read_references(directory, frames):
    manifest = json.loads((directory / "manifest.json").read_text())
    episodes = {}
    keys = ("root_pose", "root_velocity", "encoder_position", "encoder_velocity")
    for episode in manifest["episode_files"]:
        with np.load(directory / episode["file"], allow_pickle=False) as data:
            if tuple(data["joint_names"]) != JOINT_NAMES or len(data["root_pose"]) != frames + 1:
                raise ValueError("Flamingo reference does not match the rollout")
            episodes[episode["support"]] = {key: torch.from_numpy(data[key][:, 0].copy()) for key in keys}
    return {key: torch.stack([episodes[side][key] for side in ("left", "right")]).float() for key in keys}


def standing_still(measure, root_velocity):
    return (
        measure["left_contact"] and measure["right_contact"]
        and measure["tilt_deg"] < 15
        and np.linalg.norm(root_velocity[:3]) < .025
        and np.linalg.norm(root_velocity[3:]) < .25
    )


def build_future(self, offsets):
    # Use the same CPU arithmetic as the checkpoint's nominal validation.
    # CUDA only receives the finished features, including for the CPU gate.
    frame = torch.arange(self.end + 1, device="cpu")
    index = (frame[:, None] + frame.new_tensor(offsets)).clamp_max(self.end)
    root = self.ref["root_pose"].cpu()
    target = root[:, index]
    inverse = qconj(normalize(root[..., 3:]))[:, :, None].expand_as(target[..., 3:])
    vel = self.ref["root_velocity"].cpu()[:, index]
    phase = (index / self.end)[None, :, :, None].expand(2, -1, -1, -1)
    future = torch.cat((rotate(inverse, target[..., :3] - root[:, :, None, :3]),
                        quat6(qmul(inverse, target[..., 3:])),
                        rotate(inverse, vel[..., :3]), rotate(inverse, vel[..., 3:]),
                        self.ref["encoder_position"].cpu()[:, index] - self.roll.home.cpu(),
                        .1 * self.ref["encoder_velocity"].cpu()[:, index], phase), dim=-1)
    return future.flatten(2).to(self.device)


def policy_action(self, model, history, side, frame):
    previous = history[:, -1, -14:]
    history = history.flatten(1)
    future = self.future[side, frame]
    def bound(raw):
        return bounded_action(self.action_envelope * raw, 0., self.roll.home,
                              self.joint_limits, self.action_limits)
    action = bound(model(history, future))
    if hasattr(model, "landing_policy"):
        lower = bound(model.landing_policy(history, future))
        action = landing_action(action, lower, previous, frame, self.roll.bounds[3],
                                getattr(self, "landing_transition", None))
    return action
