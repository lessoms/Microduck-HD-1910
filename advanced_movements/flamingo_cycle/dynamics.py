"""Nominal vector rollout for Flamingo Cycle inference."""
import torch
from mjlab.envs import ManagerBasedRlEnv

from fall_risk import has_fallen
from environment_variations import policy_observation
from microduck_sm.environment import TrainingEnvironment
from diagnose_hd1910 import JOINTS
from record_policy_reference import configuration

JOINT_NAMES = tuple(joint["name"] for joint in JOINTS)


class Rollout:
    state = TrainingEnvironment.state

    def __init__(self, n, device, bounds, seed):
        self.n, self.device = n, device
        self.bounds = bounds
        cfg = configuration()
        cfg.scene.num_envs = n
        cfg.episode_length_s = (self.bounds[-1] + 1) / 50
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.robot = self.env.scene["robot"]
        self.d, self.a = self.robot.data, self.robot.actuators[0]
        self.ids = [self.robot.joint_names.index(name) for name in JOINT_NAMES]
        self.bl = [self.robot.joint_names.index(f"passive_{name}_backlash") for name in JOINT_NAMES]
        self.home = self.d.default_joint_pos[0, self.ids].clone()
        self.schema = None
        self.env.reset(seed=seed)

    def reset(self, episode_start):
        lanes = torch.arange(episode_start, episode_start + self.n, device=self.device)
        self.side = (lanes + lanes // 10) % 2
        self.env.reset()
        self.last = self.home.new_zeros(self.n, 14)
        self.command = self.home.new_zeros(self.n, 13)
        self.history = self.proprio()[:, None].repeat(1, 10, 1)
        self.state()
        self.alive = torch.ones(self.n, dtype=torch.bool, device=self.device)

    def proprio(self):
        return policy_observation(self.env, self.last, self.command)[:, :48]

    def step(self, action, tick):
        mask = self.alive.clone()
        action = torch.where(mask[:, None], action, 0.)
        self.env.step(action)
        self.last = action
        successor = self.state()
        if not torch.isfinite(successor[mask]).all():
            raise RuntimeError("Nonfinite HD1910 rollout state")
        fallen = has_fallen(successor)
        done = fallen | (tick + 1 == self.bounds[-1])
        self.alive &= ~done
        self.history = torch.cat((self.history[:, 1:], self.proprio()[:, None]), 1)

    def close(self):
        self.env.close()
