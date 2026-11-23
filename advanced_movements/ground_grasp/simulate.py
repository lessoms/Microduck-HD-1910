"""Configure the Microduck beak and grasp object."""
from copy import deepcopy
from pathlib import Path
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.utils.spec_config import CollisionCfg
from record_policy_reference import configuration
from diagnose_hd1910 import JOINTS
from mouth_actuator import ProtectedMouthCfg
from geometry import add_mouth, get_grasp_object_spec, MOUTH_JOINT, MOUTH_MIN_RAD

ROOT = Path(__file__).resolve().parent
BODY_NAMES = [joint['name'] for joint in JOINTS]


def build_config(object_height_m=.016):
    cfg = configuration()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 10
    robot = cfg.scene.entities['robot']
    original_spec = robot.spec_fn
    def spec_fn():
        spec = original_spec()
        add_mouth(spec)
        return spec
    robot.spec_fn = spec_fn
    robot.init_state.joint_pos = {MOUTH_JOINT: MOUTH_MIN_RAD, **robot.init_state.joint_pos}
    robot.articulation.actuators += (ProtectedMouthCfg(
        target_names_expr=(MOUTH_JOINT,), effort_limit=.5),)
    robot.collisions = (CollisionCfg(
        geom_names_expr=['.*_collision'],
        condim={r'^(left|right)_foot_collision$':3,
                r'^(upper_beak|jaw_plate)_collision$':4, '.*_collision':1},
        priority={r'^(left|right)_foot_collision$':1},
        friction={r'^(left|right)_foot_collision$':(1.,),
                  r'^(upper_beak|jaw_plate)_collision$':(1.5,.02,.0001)}),)
    cfg.scene.entities['object'] = EntityCfg(spec_fn=lambda: get_grasp_object_spec(object_height_m),
        init_state=EntityCfg.InitialStateCfg(pos=(.087,0.,object_height_m/2)))
    cfg.scene.sensors = tuple(ContactSensorCfg(
        name=f'object_{side}_beak',
        primary=ContactMatch(mode='geom',pattern=f'^{geom}$',entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='^object_geom$',entity='object'),
        fields=('found',),reduce='netforce',num_slots=1)
        for side,geom in (('upper','upper_beak_collision'),('lower','jaw_plate_collision')))
    return cfg
