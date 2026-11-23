"""Explicit motor PD with substep closing-load trip and latched low-torque hold."""
from dataclasses import dataclass
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg
from mjlab.utils.spec import create_motor_actuator
from mouth_protection import MouthTorqueGuard


class ProtectedMouthActuator(Actuator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guard=MouthTorqueGuard()
        self.trace=[];self.control_time=0.;self.substep=0

    @property
    def command_field(self):
        return 'effort'

    def edit_spec(self,spec,target_names):
        for name in target_names:
            self._mjs_actuators.append(create_motor_actuator(spec,name,
                effort_limit=self.cfg.effort_limit,gear=1.,armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,viscous_damping=self.cfg.viscous_damping,
                transmission_type=self.cfg.transmission_type))

    def initialize(self,mj_model,model,data,device):
        super().initialize(mj_model,model,data,device)
        self.physics_dt=float(mj_model.opt.timestep)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.guard.reset();self.trace=[];self.control_time=0.;self.substep=0

    def begin_step(self,time_s):
        self.control_time=time_s;self.substep=0

    def compute(self,cmd):
        if cmd.pos.numel()!=1:
            raise ValueError('This probe requires one world and one mouth motor')
        measured=float(self.entity.data.qfrc_actuator[:,self._target_ids_list].item())
        row=self.guard.update(float(cmd.position_target.item()),float(cmd.pos.item()),
            float(cmd.vel.item()),measured,self.control_time+self.substep*self.physics_dt)
        self.trace.append(row);self.substep+=1
        return torch.full_like(cmd.pos,row['output_nm'])


@dataclass(kw_only=True)
class ProtectedMouthCfg(ActuatorCfg):
    effort_limit: float = .5
    def build(self,entity,target_ids,target_names):
        return ProtectedMouthActuator(self,entity,target_ids,target_names)
