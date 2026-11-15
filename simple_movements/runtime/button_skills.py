"""Keyboard-triggered trained skills in continuous Microduck MuJoCo simulation."""
import argparse
import copy
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
import threading
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_sm.environment import TrainingEnvironment
from microduck_sm.math_fk import qmul, rotate
from microduck_sm.policy import OldAlignedPolicy
from command_encoding import condition
from environment_variations import bounded_action

DEFAULT_SIT = "models/sitstand_0120.pt"
DEFAULT_VELOCITY = "models/velocity_0470.pt"
DEFAULT_ROLL = "models/roulade_0380.pt"


def yaw(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


class ButtonController:
    def __init__(self, sit_checkpoint, roll_checkpoint, seed, velocity_checkpoint=DEFAULT_VELOCITY):
        self.policies = {}
        self.configs = {}
        self.references = {}
        for name, path in (("sitstand", sit_checkpoint), ("roll", roll_checkpoint), ("velocity",velocity_checkpoint)):
            saved = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            if saved.get("architecture") != ("old_aligned_velocity_v1" if name=="velocity" else "old_aligned_v1"):
                raise ValueError(f"Unsupported checkpoint architecture in {path}")
            config = copy.deepcopy(saved["collection_config"])
            if config["history_frames"] != 10:
                raise ValueError("Button skills require ten observation frames")
            config.update(num_envs=1, seed=seed, max_episode_steps=10**9)
            policy = OldAlignedPolicy().cuda().eval().requires_grad_(False)
            policy.load_state_dict(saved["models"]["policy"])
            self.policies[name] = policy
            self.configs[name] = config
            with np.load(ROOT / config["reference"]) as archive:
                self.references[name] = {
                    key: torch.as_tensor(archive[key], device="cuda")
                    for key in archive.files
                    if np.issubdtype(archive[key].dtype, np.number) or archive[key].dtype == bool
                }
            del saved
        self.env = TrainingEnvironment(self.configs["sitstand"])
        self.active = "sitstand"
        self.state = "standing"
        self.events = []
        self.ticks = 0
        self.reset_count = 0
        self.stable_ticks = 0
        self.idle_ticks = 0
        self.motion_ticks = 0
        self.blend_ticks = 0
        self.blend_start = None
        self.requested_velocity=torch.zeros((1,3),device="cuda")
        self.applied_velocity=torch.zeros_like(self.requested_velocity)
        self.pending_skill=None
        self.blend_total=10
        self.velocity_peak=0.
        self.reset()

    @torch.inference_mode()
    def activate(self, name, start, align=True):
        env = self.env
        reference = dict(self.references[name])
        if align and name!="velocity":
            current = env.d.root_link_pose_w[0].clone()
            current[:3] -= env.env.scene.env_origins[0]
            anchor = reference["root_pose"][start, 0].float()
            angle = yaw(current[3:]) - yaw(anchor[3:])
            rotation = torch.stack((torch.cos(angle / 2), angle * 0, angle * 0, torch.sin(angle / 2)))
            shift = current[:3] - rotate(rotation, anchor[:3])
            shift[2] = 0

            def transform_pose(value):
                value = value.float().clone()
                q = rotation.expand(*value.shape[:-1], 4)
                value[..., :3] = rotate(q, value[..., :3]) + shift
                value[..., 3:7] = qmul(q, value[..., 3:7])
                return value

            for key in ("root_pose", "body_pose", "qpos"):
                reference[key] = transform_pose(reference[key])
            for key in ("root_velocity", "body_velocity", "qvel"):
                value = reference[key].float().clone()
                q = rotation.expand(*value.shape[:-1], 4)
                value[..., :3] = rotate(q, value[..., :3])
                value[..., 3:6] = rotate(q, value[..., 3:6])
                reference[key] = value
        env.ref = reference
        env.cfg = copy.deepcopy(self.configs[name])
        env.reference_end = env.cfg["reference_end"]
        env.offsets = torch.tensor(env.cfg["future_offsets"], device=env.device)
        env.speed_weights = None
        if env.cfg.get("speed_penalty"):
            env.speed_weights = env.reference_speed_weights(
                env.ref["root_velocity"][:env.reference_end + 1, 0].float()
            )
        env.frame.fill_(start)
        self.active = name
        self.blend_start = env.last.clone()
        self.blend_ticks = self.blend_total = 10

    @torch.inference_mode()
    def reset(self):
        self.activate("sitstand", 0, align=False)
        ids = torch.tensor([0], device=self.env.device)
        self.env.reset(ids, torch.zeros_like(ids))
        self.state = "standing"
        self.hold_frame = 25
        self.stable_ticks = self.idle_ticks = self.motion_ticks = 0
        self.blend_ticks = 0
        self.requested_velocity.zero_();self.applied_velocity.zero_();self.pending_skill=None
        self.reset_count += 1
        self.record("reset")

    def record(self, event, **details):
        item = dict(time_s=round(self.ticks * .02, 3), event=event, state=self.state, **details)
        self.events.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def posture_stable(self, seated=False):
        env = self.env
        height = float(env.d.root_link_pos_w[0, 2] - env.env.scene.env_origins[0, 2])
        velocity = env.d.root_link_vel_w[0]
        up = float(-env.d.projected_gravity_b[0, 2])
        quiet = float(velocity[:3].norm()) < .03 and float(velocity[3:].norm()) < .3
        if seated:
            return height < .075 and quiet
        return height > .10 and up > math.cos(math.radians(10)) and quiet

    @torch.inference_mode()
    def set_velocity(self, vx, wz):
        vx=float(np.clip(vx,-.22,.22));wz=float(np.clip(wz,-.75,.75))
        if not np.isfinite([vx,wz]).all():raise ValueError("Velocity commands must be finite")
        if self.state=="standing" and (vx!=0 or wz!=0):
            if self.stable_ticks<8:
                self.record("rejected",command="velocity",reason="等待站立稳定");return False
            self.activate("velocity",0,align=False)
            self.state="velocity_preparing";self.velocity_prepare_ticks=0
            self.blend_ticks=self.blend_total=50;self.pending_skill=None
            self.applied_velocity.zero_();self.velocity_peak=0.
            self.velocity_origin=self.env.d.root_link_pos_w[0,:2].clone()
            self.record("velocity_enter")
        if self.state not in ("velocity_preparing","walking","velocity_stopping"):
            if vx==0 and wz==0:return True
            self.record("rejected",command="velocity",reason="需先起立并完成当前动作");return False
        if self.pending_skill is not None and (vx!=0 or wz!=0):
            self.record("rejected",command="velocity",reason="正在停车切换动作");return False
        self.requested_velocity[:]=self.requested_velocity.new_tensor([[vx,0.,wz]])
        if vx==0 and wz==0:
            if self.state!="velocity_stopping":self.stop_ticks=0
            self.state="velocity_stopping"
        elif self.state=="velocity_stopping":self.state="walking"
        self.record("velocity_command",vx=vx,wz=wz)
        return True

    @torch.inference_mode()
    def velocity_step(self):
        env=self.env
        self.velocity_prepare_ticks+=1
        target=self.requested_velocity if self.velocity_prepare_ticks>=100 else torch.zeros_like(self.requested_velocity)
        limits=self.applied_velocity.new_tensor([[.0088,0.,.03]])
        self.applied_velocity+=(target-self.applied_velocity).clamp(-limits,limits)
        raw=self.policies["velocity"](env.history.flatten(1),condition(self.applied_velocity))
        action=env.executed_action(raw,torch.zeros_like(raw))
        if self.blend_ticks:
            fraction=(self.blend_total-self.blend_ticks+1)/self.blend_total
            action=self.blend_start.lerp(action,fraction);self.blend_ticks-=1
        # One shared physics environment. Walking has no fixed-path termination.
        env.env.step(action);env.last=action.clone();env.age+=1
        env.frame=(env.frame+1).clamp_max(env.reference_end)
        env.history=torch.cat((env.history[:,1:],env.proprio()[:,None]),1)
        state=env.state();pose=torch.cat((state[:,:7],state[:,13:41]),1)
        if env.pose_history is not None:env.pose_history=torch.cat((env.pose_history[:,1:],pose[:,None]),1)
        up=1-2*(state[:,4]**2+state[:,5]**2);enc=env.d.joint_pos[:,env.ids]+env.d.joint_pos[:,env.bl];lim=env.d.default_joint_pos_limits[:,env.ids]
        violation=torch.maximum(lim[:,:,0]-enc,enc-lim[:,:,1]).clamp_min(0).amax(1)
        reason=int(up[0]<math.cos(math.radians(55)))+2*int(state[0,2]<.07)+4*int(violation[0]>.1)+8*int(not torch.isfinite(state).all())
        self.ticks+=1;stable=self.posture_stable();self.stable_ticks=self.stable_ticks+1 if stable else 0
        self.velocity_peak=max(self.velocity_peak,float(env.d.root_link_lin_vel_b[0,:2].norm()))
        if reason:
            self.state="fault";self.record("fault",reason_bits=reason,source="velocity");return
        if self.state=="velocity_preparing" and self.velocity_prepare_ticks>=100:self.state="walking";self.record("velocity_ready")
        if self.state=="velocity_stopping":
            self.stop_ticks+=1
            if self.stop_ticks>=100 and self.stable_ticks>=50 and self.applied_velocity.abs().max()<1e-5 and self.velocity_prepare_ticks>=100 and float((env.d.joint_vel[0,env.ids]+env.d.joint_vel[0,env.bl]).square().mean().sqrt())<.3:
                distance=float((env.d.root_link_pos_w[0,:2]-self.velocity_origin).norm())
                self.record("velocity_stopped",distance_m=distance,peak_planar_speed=self.velocity_peak)
                self.activate("sitstand",25);self.blend_ticks=self.blend_total=100
                self.state="skill_preparing";self.prepare_ticks=0;self.stable_ticks=0;self.hold_frame=25
            elif self.stop_ticks>500:
                self.state="fault";self.record("fault",reason="速度策略归零后未稳定，拒绝强行切换")

    @torch.inference_mode()
    def command(self, command):
        if command == "reset":
            self.reset()
            return True
        if command in ("forward","backward","left","right","stop"):
            v=float(self.requested_velocity[0,0]);w=float(self.requested_velocity[0,2])
            return self.set_velocity(v+.025 if command=="forward" else v-.025 if command=="backward" else 0. if command=="stop" else v,
                                     w+.10 if command=="left" else w-.10 if command=="right" else 0. if command=="stop" else w)
        if self.state in ("walking","velocity_preparing","velocity_stopping") and command in ("sit","roll"):
            self.pending_skill=command;self.set_velocity(0.,0.);self.record("queued_after_stop",command=command);return True
        allowed = {"sit": "standing", "stand": "sitting", "roll": "standing"}
        if self.state != allowed.get(command) or self.stable_ticks < 8:
            self.record("rejected", command=command, reason="等待当前动作完成并稳定")
            return False
        if command == "sit":
            self.activate("sitstand", 100)
            self.end_frame = 200
            self.state = "sitting_down"
        elif command == "stand":
            self.activate("sitstand", 260)
            self.end_frame = self.env.reference_end
            self.state = "standing_up"
        else:
            self.activate("roll", 25)
            self.end_frame = self.env.reference_end
            self.state = "roll_preparing"
            self.prepare_ticks = 0
        self.max_motion_ticks = self.end_frame - int(self.env.frame[0]) + 250
        self.stable_ticks = self.idle_ticks = self.motion_ticks = 0
        self.record("accepted", command=command)
        return True

    @torch.inference_mode()
    def step(self):
        if self.state == "fault":
            return
        if self.state in ("walking","velocity_preparing","velocity_stopping"):
            self.velocity_step();return
        env = self.env
        idle = self.state in ("standing", "sitting","skill_preparing")
        if self.state == "roll_preparing":
            env.frame.fill_(25)
        elif idle:
            env.frame.fill_(self.hold_frame)
        else:
            env.frame.clamp_(max=self.end_frame)
        raw = self.policies[self.active](env.history.flatten(1), env.future())
        action = env.executed_action(raw, torch.zeros_like(raw))
        if self.blend_ticks:
            fraction = (self.blend_total - self.blend_ticks + 1) / self.blend_total
            if self.state=="skill_preparing":
                source=self.policies["velocity"](env.history.flatten(1),condition(torch.zeros_like(self.applied_velocity)))
                source=bounded_action(source,0.,env.d.default_joint_pos[:,env.ids],env.d.default_joint_pos_limits[:,env.ids],self.configs["velocity"])
                action=source.lerp(action,fraction)
            else:
                action = self.blend_start.lerp(action, fraction)
            self.blend_ticks -= 1
        _, _, terminated, _, reason = env.step(action)
        self.ticks += 1
        if bool(terminated[0]):
            self.state = "fault"
            self.record(
                "fault", reason_bits=int(reason[0]), reference_frame=int(env.frame[0]),
                height_m=float(env.d.root_link_pos_w[0, 2] - env.env.scene.env_origins[0, 2]),
                linear_speed_m_s=float(env.d.root_link_vel_w[0, :3].norm()),
                angular_speed_rad_s=float(env.d.root_link_vel_w[0, 3:].norm()),
            )
            return
        seated = self.state in ("sitting", "sitting_down")
        stable = self.posture_stable(seated)
        self.stable_ticks = self.stable_ticks + 1 if stable else 0
        if self.state=="skill_preparing":
            self.prepare_ticks+=1
            enc=env.d.joint_pos[0,env.ids]+env.d.joint_pos[0,env.bl]
            error=float((enc-env.ref["encoder_position"][25,0]).square().mean().sqrt())
            if self.prepare_ticks>=150 and self.blend_ticks==0 and self.stable_ticks>=50 and error<.06 and float((env.d.joint_vel[0,env.ids]+env.d.joint_vel[0,env.bl]).square().mean().sqrt())<.2:
                pending=self.pending_skill;self.pending_skill=None;self.state="standing";self.idle_ticks=0
                self.record("handoff_complete",joint_pose_rms_rad=error)
                if pending:self.command(pending)
            elif self.prepare_ticks>400:
                self.state="fault";self.record("fault",reason="步行到技能姿态交接未稳定")
            return
        if idle:
            self.idle_ticks += 1
            return
        self.motion_ticks += 1
        if self.state == "roll_preparing":
            self.prepare_ticks += 1
            enc = env.d.joint_pos[0, env.ids] + env.d.joint_pos[0, env.bl]
            target = env.ref["encoder_position"][25, 0].float()
            pose_error = float((enc - target).square().mean().sqrt())
            if self.prepare_ticks >= 100 and self.stable_ticks >= 50 and pose_error < .06 and float((env.d.joint_vel[0,env.ids]+env.d.joint_vel[0,env.bl]).square().mean().sqrt())<.2:
                self.state = "rolling"
                env.frame.fill_(75)
                self.max_motion_ticks = self.end_frame - 75 + 250
                self.motion_ticks = 0
                self.record("prepared", joint_pose_rms_rad=pose_error)
            elif self.prepare_ticks >= 250:
                self.state = "fault"
                self.record("fault", reason="前滚翻准备姿态未稳定")
            return
        if int(env.frame[0]) >= self.end_frame:
            if self.stable_ticks >= 8:
                completed = self.state
                self.state = "sitting" if seated else "standing"
                self.hold_frame = self.end_frame
                self.idle_ticks = 0
                self.record("completed", motion=completed)
            elif self.motion_ticks > self.max_motion_ticks:
                self.state = "fault"
                self.record("fault", reason="动作末段未达到稳定姿态")

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitstand-checkpoint", default=DEFAULT_SIT)
    parser.add_argument("--roll-checkpoint", default=DEFAULT_ROLL)
    parser.add_argument("--velocity-checkpoint", default=DEFAULT_VELOCITY)
    parser.add_argument("--demo-combined",action="store_true",help="Test sit/stand, forward/stop and roll continuously")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Automatically press sit, stand, roll once")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.speed <= 0 or (args.max_seconds is not None and args.max_seconds <= 0):
        parser.error("Speed and duration must be positive")
    if args.headless and not (args.demo or args.demo_combined) and args.max_seconds is None:
        parser.error("Headless execution requires --demo or --max-seconds")

    controller = None
    viewer = None
    viewer_threads = []
    commands = SimpleQueue()
    quit_requested = False

    def key_callback(code):
        commands.put({83:"sit",85:"stand",70:"roll",82:"reset",81:"quit",87:"forward",88:"backward",65:"left",68:"right",265:"forward",264:"backward",263:"left",262:"right",32:"stop"}.get(code, ""))

    try:
        torch.set_float32_matmul_precision("high")
        controller = ButtonController(args.sitstand_checkpoint, args.roll_checkpoint, args.seed,args.velocity_checkpoint)
        if not args.headless:
            import mujoco.viewer
            model=copy.copy(controller.env.env.sim.mj_model)
            display_data=mujoco.MjData(model)
            names=controller.env.robot.joint_names
            addresses=[model.jnt_qposadr[model.joint("robot/"+name).id] for name in names]
            existing_threads = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(model, display_data, key_callback=key_callback)
            viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
            viewer.cam.distance = .75
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        print("W/X或上下箭头 前后 | A/D或左右箭头 转向 | 空格停车 | S坐下 U起立 F前滚翻 | R重置 Q退出。指令松键保留，前进建议累计到0.15、转向0.5测试。",flush=True)
        auto=args.demo or args.demo_combined
        demo=["sit","stand","drive","stop","roll"] if args.demo_combined else ["sit","stand","roll"] if args.demo else []
        drive_until=0
        max_seconds = args.max_seconds or (90 if args.demo_combined else 45 if args.demo else float("inf"))
        while not quit_requested and controller.ticks * .02 < max_seconds:
            start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            try:
                while True:
                    command = commands.get_nowait()
                    if command == "quit":
                        quit_requested = True
                    elif command:
                        controller.command(command)
            except Empty:
                pass
            if demo and demo[0]=="stop" and controller.state=="walking" and controller.ticks>=drive_until:
                controller.command("stop");demo.pop(0)
            if demo and demo[0]=="drive" and controller.state=="standing" and controller.idle_ticks>=50 and controller.stable_ticks>=8:
                if controller.set_velocity(.15,0.):demo.pop(0);drive_until=controller.ticks+200
            if demo and demo[0] not in ("drive","stop") and controller.state in ("standing","sitting") and controller.idle_ticks >= 50 and controller.stable_ticks >= 8:
                if controller.command(demo[0]):
                    demo.pop(0)
            controller.step()
            if viewer is not None:
                state = controller.env.state()[0].cpu().numpy()
                with viewer.lock():
                    display_data.qpos[:7] = state[:7]
                    display_data.qpos[addresses] = state[13:41]
                    mujoco.mj_forward(model, display_data)
                    viewer.cam.lookat[:] = state[:3]
                viewer.sync()
            if controller.state == "fault" and args.headless:
                break
            if auto and not demo and controller.state == "standing" and controller.idle_ticks >= 200:
                break
            if not args.headless:
                time.sleep(max(0, .02 / args.speed - (time.monotonic() - start)))
        complete = auto and not demo and controller.state == "standing" and controller.idle_ticks >= 200
        result = dict(state=controller.state, elapsed_s=controller.ticks * .02,
                      reset_count=controller.reset_count, demo_complete=complete,
                      events=controller.events)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if auto and not complete:
            raise RuntimeError("Button demo did not complete; inspect the events/report")
    finally:
        if viewer is not None:
            import glfw
            viewer.close()
            glfw.post_empty_event()
            for thread in viewer_threads:
                thread.join(timeout=2)
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
