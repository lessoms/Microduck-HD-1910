"""Per-world, partial-reset-safe torch mouth protection, evaluated every 5ms."""
from dataclasses import dataclass
import torch
from mjlab.actuator.actuator import Actuator,ActuatorCfg
from mjlab.utils.spec import create_motor_actuator
from geometry import MOUTH_MIN_RAD,MOUTH_MAX_RAD
from mouth_protection import TRIP_RATIO,HOLD_RATIO

class VectorMouth(Actuator):
 @property
 def command_field(self):return 'effort'
 def edit_spec(self,spec,target_names):
  for name in target_names:self._mjs_actuators.append(create_motor_actuator(spec,name,effort_limit=.5,gear=1.,armature=None,frictionloss=None,viscous_damping=None,transmission_type=self.cfg.transmission_type))
 def initialize(self,mj_model,model,data,device):
  super().initialize(mj_model,model,data,device)
  n=data.nworld
  self.baseline=torch.full((n,1),self.cfg.baseline_nm,device=device)
  self.armed=torch.zeros(n,1,device=device,dtype=torch.bool)
  self.latched=self.armed.clone();self.frozen=torch.full((n,1),MOUTH_MIN_RAD,device=device)
  self.last_torque=torch.zeros(n,1,device=device)
  self.output_peak=torch.zeros(n,1,device=device)
 def reset(self,env_ids=None):
  super().reset(env_ids)
  if not hasattr(self,'latched'):return
  ids=slice(None) if env_ids is None else env_ids
  self.armed[ids]=False;self.latched[ids]=False;self.frozen[ids]=MOUTH_MIN_RAD
  self.last_torque[ids]=0.;self.output_peak[ids]=0.
 def compute(self,cmd):
  target=cmd.position_target.clamp(MOUTH_MIN_RAD,MOUTH_MAX_RAD)
  demand=(target-cmd.pos)-.01*cmd.vel
  measured=self.entity.data.qfrc_actuator[:,self.target_ids]
  trip=self.armed&~self.latched&(torch.maximum(-demand,-measured)>=TRIP_RATIO*self.baseline)
  self.frozen=torch.where(trip,cmd.pos.clamp(MOUTH_MIN_RAD,MOUTH_MAX_RAD),self.frozen)
  self.latched|=trip
  limit=(TRIP_RATIO*self.baseline).clamp_max(.5)
  torque=demand.clamp(-limit,limit)
  torque=torch.where(self.latched,-HOLD_RATIO*self.baseline,torque)
  torque=torch.where(((cmd.pos<=MOUTH_MIN_RAD)&(torque<0))|((cmd.pos>=MOUTH_MAX_RAD)&(torque>0)),0.,torque)
  self.last_torque=torque.clone();self.output_peak=torch.maximum(self.output_peak,torque.abs())
  return torque

@dataclass(kw_only=True)
class VectorMouthCfg(ActuatorCfg):
 baseline_nm:float=.08672039955854416
 def __post_init__(self):
  super().__post_init__()
  if not 0<self.baseline_nm<=.5/TRIP_RATIO:raise ValueError('Unsafe baseline')
 def build(self,entity,target_ids,target_names):return VectorMouth(self,entity,target_ids,target_names)
