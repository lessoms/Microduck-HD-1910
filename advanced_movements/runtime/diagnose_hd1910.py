"""Build the nominal HD-1910 control environment."""
import json
from dataclasses import replace
from pathlib import Path
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.envs.mdp.events import reset_scene_to_default

ROOT = Path(__file__).resolve().parents[2] / 'simple_movements'
JOINTS = json.loads((ROOT / 'configs/robot_audit.json').read_text())['joints']
SCALE = json.loads((ROOT / "configs/robot_baseline.json").read_text())["action_scale_rad"]
MARGIN = 0.02


def reset_to_home(env, env_ids):
    reset_scene_to_default(env, env_ids)
    robot = env.scene['robot']
    # Seed targets before reset() writes to simulation and fills the delay queue.
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)


def make_cfg(fixed=False):
    cfg = load_env_cfg('Mjlab-Velocity-Flat-MicroDuck-HD1910')
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.seed = 7
    robot = cfg.scene.entities['robot']
    robot.init_state.pos = (0., 0., 0.4 if fixed else 0.12)
    if fixed:
        cfg.sim.mujoco.gravity = (0., 0., 0.)
    robot.articulation.actuators = tuple(replace(a, target_names_expr=tuple(r['name'] for r in JOINTS),
        vin_range=(7.4,7.4), vin_drop_gain_range=(0.1,0.1), delay_min_lag=4,delay_max_lag=4)
        for a in robot.articulation.actuators)
    if fixed:
        original = robot.spec_fn
        def fixed_spec():
            spec = original()
            spec.delete(spec.joint('trunk_base_freejoint'))
            return spec
        robot.spec_fn = fixed_spec
    cfg.events = {'default_reset': EventTermCfg(func=reset_to_home,mode='reset'),
                  'bam_fields':cfg.events['expand_bam_friction_fields']}
    cfg.observations = {}
    cfg.commands = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.scene.sensors = ()
    action = cfg.actions['joint_pos']
    action.scale = SCALE
    action.actuator_names = tuple(r['name'] for r in JOINTS)
    action.preserve_order = True
    action.clip = {r['name']:(r['limits_rad'][0]+MARGIN,r['limits_rad'][1]-MARGIN) for r in JOINTS}
    cfg.viewer.distance = 0.65
    cfg.viewer.elevation = -15
    cfg.viewer.azimuth = 135
    cfg.viewer.width = 640
    cfg.viewer.height = 480
    return cfg
