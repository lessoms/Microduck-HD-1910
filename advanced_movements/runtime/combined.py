"""Run the three advanced MuJoCo skills from one keyboard-controlled window."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("kick", "grasp", "flamingo")
LABELS = ("Right-foot kick", "Mouth grasp", "Right-foot balance")
DIRECTORIES = dict(zip(SKILLS, ("kick_ball", "ground_grasp", "flamingo_cycle")))


@contextmanager
def task_imports(skill):
    """Load each task's local modules, including its own simulate.py."""
    previous = sys.modules.pop("simulate", None)
    sys.path.insert(0, str(ROOT / DIRECTORIES[skill]))
    try:
        yield
    finally:
        sys.path.pop(0)
        sys.modules.pop("simulate", None)
        if previous is not None:
            sys.modules["simulate"] = previous


def config_for(skill):
    return json.loads((ROOT / "configs" / f"{skill}.json").read_text())


def load_weights(actor, path, device):
    weights = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    actor.load_state_dict(weights)
    return actor.to(device).eval()


def check_schema(actual, expected):
    if {key: list(bounds) for key, bounds in actual.items()} != expected:
        raise ValueError("Reference and simulation state layouts differ")


class Kick:
    distance = .85

    def __init__(self, args):
        from aligned import TrackingEnvironment, TrackingPolicy

        checkpoint = args.kick_checkpoint or ROOT / "models/kick_0420.pt"
        saved = config_for("kick")
        config = dict(saved["environment"], device=args.device, num_envs=2,
                      reference=str(ROOT / "references/kick.npz"))
        self.actor = load_weights(TrackingPolicy(saved["policy"]), checkpoint, args.device)
        self.env = TrackingEnvironment(config)
        self.sim = self.env.env.sim
        self.ids = torch.arange(self.env.n, device=args.device)
        self.frames = torch.zeros_like(self.ids)
        self.zero_action = torch.zeros(self.env.n, 14, device=args.device)

    def reset(self):
        self.env.generator.manual_seed(20261117)
        self.env.reset(self.ids, self.frames)
        self.tick = 0

    def step(self):
        action = self.actor(self.env.history.flatten(1), self.env.future())
        _, _, terminated, truncated, _ = self.env.step(
            self.env.executed_action(action, self.zero_action)
        )
        self.tick += 1
        return bool(terminated[0] | truncated[0]) or self.tick >= self.env.reference_end

    def result(self):
        distance = self.env.ball.data.root_link_pos_w[0, 0] - self.env.ball_start[0, 0]
        contact = bool(self.env.kick_right_touch[0])
        return (f"{self.tick} steps, right-foot contact={contact}, "
                f"forward ball displacement {float(distance):.3f} m")

    def close(self):
        self.env.close()


class Grasp:
    distance = .55

    def __init__(self, args):
        from environment_variations import bounded_action
        from microduck_sm.old_aligned import OldAlignedPolicy
        from training_env import GraspEnvironment

        checkpoint = args.grasp_checkpoint or ROOT / "models/grasp_0120.pt"
        saved = config_for("grasp")
        with np.load(ROOT / "references/grasp.npz", allow_pickle=False) as arrays:
            reference = {key: torch.from_numpy(arrays[key]) for key in ("reference", "initial_context")}
        self.actor = load_weights(OldAlignedPolicy(), checkpoint, args.device)
        self.bound = bounded_action
        self.env = GraspEnvironment(saved["spec"], 1, args.device)
        self.ids = torch.zeros(1, dtype=torch.long, device=args.device)
        try:
            self.env.state()
            check_schema(self.env.schema, saved["schema"])
            self.env.reference = reference["reference"].to(args.device)
            self.initial_context = reference["initial_context"].to(args.device)
            self.sim = self.env.env.sim
        except BaseException:
            self.env.close()
            raise

    def reset(self):
        self.env.restore(self.env.reference[:1], self.initial_context, self.ids)
        self.tick = 0

    def step(self):
        env = self.env
        end = env.cfg["reference_end"]
        frame = (env.clock[:, 0] * 50).round().long().clamp_max(end)
        raw = self.actor(env.history.flatten(1), env.future(frame))
        action = self.bound(raw, 0., env.home, env.limits, env.cfg)
        _, _, _, done = env.step(action)
        self.tick += 1
        return bool(done[0]) or self.tick >= end

    def result(self):
        return (f"{self.tick} steps, success={bool(self.env.success[0, 0])}, "
                f"peak lift {100 * float(self.env.peak[0, 0]):.2f} cm")

    def close(self):
        self.env.close()


class Flamingo:
    distance = .65

    def __init__(self, args):
        from control import build_future, policy_action, read_references, standing_still
        from dynamics import Rollout
        from microduck_sm.old_aligned import OldAlignedPolicy
        from simulate import FootMeasurements

        checkpoint = args.flamingo_checkpoint or ROOT / "models/flamingo_0290.pt"
        saved = config_for("flamingo")
        self.policy_action = policy_action
        self.standing_still = standing_still
        weights = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        self.actor = OldAlignedPolicy()
        self.actor.landing_policy = OldAlignedPolicy()
        self.actor.load_state_dict(weights)
        self.actor.to(args.device).eval()
        self.rollout = Rollout(2, args.device, saved["rollout"]["phase_bounds"], 20261201)
        try:
            self.rollout.reset(1)
            check_schema(self.rollout.schema, saved["schema"])
            roll = self.rollout
            end = roll.bounds[-1]
            references = read_references(ROOT / "references/flamingo", end)
            future = build_future(SimpleNamespace(roll=roll, end=end, ref=references, device=roll.device),
                                  saved["rollout"]["future_offsets"])
            self.view = SimpleNamespace(roll=roll, future=future,
                                        action_limits=saved["action_limits"],
                                        action_envelope=saved["action_envelope_rad"],
                                        joint_limits=roll.d.default_joint_pos_limits[0, roll.ids],
                                        landing_transition=saved["landing_transition"])
            self.sim = self.rollout.env.sim
            self.feet = FootMeasurements(self.sim.mj_model)
        except BaseException:
            self.rollout.close()
            raise

    def reset(self):
        # Odd episode 1 puts right-foot support in the displayed lane 0.
        self.rollout.reset(1)
        self.tick = self.held = self.recovered = 0

    def step(self):
        roll = self.rollout
        frame = torch.full_like(roll.side, self.tick)
        action = self.policy_action(self.view, self.actor, roll.history, roll.side, frame)
        roll.step(action, self.tick)
        if self.tick >= roll.bounds[2]:
            measured = self.feet.sample(self.sim.data.qpos[0].cpu().numpy())
            if self.tick < roll.bounds[3]:
                self.held += int(measured["right_contact"] and not measured["left_contact"]
                                 and measured["left_clearance_m"] >= .05)
            else:
                velocity = roll.d.root_link_vel_w[0].cpu().numpy()
                self.recovered = self.recovered + 1 if self.standing_still(measured, velocity) else 0
        self.tick += 1
        return not bool(roll.alive[0])

    def result(self):
        return (f"{self.tick} steps, right-foot hold {self.held / 50:.2f} s, "
                f"final two-foot stability {self.recovered / 50:.2f} s")

    def close(self):
        self.rollout.close()


FACTORIES = dict(zip(SKILLS, (Kick, Grasp, Flamingo)))


class Window:
    def __init__(self, initial_skill):
        import glfw

        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError("Cannot initialize GLFW. Use --headless without a desktop.")
        self.handle = glfw.create_window(1024, 720, "Advanced Movements", None, None)
        if self.handle is None:
            glfw.terminate()
            raise RuntimeError("Cannot open the MuJoCo window")
        glfw.make_context_current(self.handle)
        glfw.swap_interval(1)
        self.requested = initial_skill
        self.replay = False
        self.context = None
        glfw.set_key_callback(self.handle, self.on_key)

    def on_key(self, window, key, scancode, action, mods):
        glfw = self.glfw
        if action != glfw.PRESS:
            return
        if glfw.KEY_1 <= key <= glfw.KEY_3:
            self.requested = SKILLS[key - glfw.KEY_1]
            self.replay = True
        elif key == glfw.KEY_R:
            self.replay = True
        elif key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)

    def bind(self, runner):
        self.release()
        self.model = runner.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = runner.distance
        self.camera.azimuth = 135
        self.camera.elevation = -18
        self.options = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def release(self):
        if self.context is not None:
            self.context.free()
            self.context = None
            self.scene = self.data = self.model = None

    def draw(self, runner, skill, finished):
        self.data.qpos[:] = runner.sim.data.qpos[0].cpu().numpy()
        self.data.qvel[:] = runner.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.qpos[:3]
        mujoco.mjv_updateScene(self.model, self.data, self.options, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        rect = mujoco.MjrRect(0, 0, *self.glfw.get_framebuffer_size(self.handle))
        mujoco.mjr_render(rect, self.scene, self.context)
        label = LABELS[SKILLS.index(skill)]
        state = "Finished" if finished else f"{runner.tick / 50:.2f} s"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           rect, f"{label} | {state}",
                           "1 Kick   2 Grasp   3 Balance   R Replay   Q Quit", self.context)
        self.glfw.swap_buffers(self.handle)

    def close(self):
        self.release()
        self.glfw.destroy_window(self.handle)
        self.glfw.terminate()


def load(skill, args):
    with task_imports(skill):
        return FACTORIES[skill](args)


def headless(args):
    skills = SKILLS if args.skill is None else (args.skill,)
    for skill in skills:
        runner = load(skill, args)
        try:
            runner.reset()
            while not runner.step():
                if args.max_steps and runner.tick >= args.max_steps:
                    break
            print(f"{skill}: {runner.result()}", flush=True)
        finally:
            runner.close()


def interactive(args):
    window = Window(args.skill or "kick")
    runner = None
    active = None
    finished = False
    try:
        while not window.glfw.window_should_close(window.handle):
            started = time.monotonic()
            window.glfw.poll_events()
            if window.glfw.window_should_close(window.handle):
                break
            if window.requested != active:
                window.release()
                if runner is not None:
                    runner.close()
                    runner = None
                active = window.requested
                runner = load(active, args)
                window.bind(runner)
                window.replay = True
            if window.replay:
                runner.reset()
                window.replay = False
                finished = False
            if not finished:
                finished = runner.step()
                if finished:
                    print(f"{active}: {runner.result()}", flush=True)
            window.draw(runner, active, finished)
            time.sleep(max(0., .02 - (time.monotonic() - started)))
    finally:
        window.close()
        if runner is not None:
            runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=SKILLS, help="Initial skill or single headless skill")
    parser.add_argument("--device", default="cpu", help="Simulation and inference device")
    parser.add_argument("--headless", action="store_true", help="Run each selected skill once")
    parser.add_argument("--max-steps", type=int, default=0, help="Headless step limit per skill")
    parser.add_argument("--kick-checkpoint", type=Path)
    parser.add_argument("--grasp-checkpoint", type=Path,
                        help="Override the default trained grasp student checkpoint")
    parser.add_argument("--flamingo-checkpoint", type=Path)
    args = parser.parse_args()
    if args.max_steps < 0 or (args.max_steps and not args.headless):
        parser.error("--max-steps requires --headless and a nonnegative value")
    for skill in SKILLS:
        checkpoint = getattr(args, f"{skill}_checkpoint")
        if checkpoint is not None:
            setattr(args, f"{skill}_checkpoint", checkpoint.expanduser().resolve())
    torch.set_float32_matmul_precision("high")
    with torch.no_grad():
        if args.headless:
            headless(args)
        else:
            interactive(args)


if __name__ == "__main__":
    main()
