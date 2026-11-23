"""Build the backlash-aware HD-1910 control scene."""
from copy import deepcopy
from dataclasses import dataclass
from diagnose_hd1910 import make_cfg, JOINTS
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.actuator.feetech_bam import FeetechBamActuator, FeetechBamActuatorCfg
from mjlab_microduck.actuator.friction_dr_bam import BacklashEncoderBamActuator


class FeetechBacklashActuator(FeetechBamActuator,BacklashEncoderBamActuator):
    """Keep Feetech reset initialization and encoder-through-backlash feedback."""


@dataclass(kw_only=True)
class FeetechBacklashCfg(FeetechBamActuatorCfg):
    def build(self,entity,target_ids,target_names):
        return FeetechBacklashActuator(self,entity,target_ids,target_names)


def configuration():
    cfg=make_cfg()
    old=cfg.scene.entities['robot']
    robot=deepcopy(MICRODUCK_BACKLASH_ROBOT_CFG)
    robot.init_state.pos=(0.,0.,.12)
    robot.articulation.actuators=(FeetechBacklashCfg(json_path=old.articulation.actuators[0].json_path,
        target_names_expr=tuple(j['name'] for j in JOINTS),kp_fw=5.,vin_range=(7.4,7.4),
        vin_drop_gain_range=(.1,.1),vin_min=7.,delay_min_lag=4,delay_max_lag=4),)
    cfg.scene.entities['robot']=robot
    cfg.actions['joint_pos'].scale=1.
    # Match the teacher's native exported position interface, including overshoot.
    cfg.actions['joint_pos'].clip=None
    return cfg
