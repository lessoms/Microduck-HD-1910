"""Explicit episode parameters only. No sampler, rollout loop, or random draws."""
import json
from pathlib import Path
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from robot_config import ROOT,JOINTS,reset_to_home

SPEC=json.loads((ROOT/'configs/domain_randomization.json').read_text())

def bounded_action(policy, noise, home, limits, config):
    """Map normalized policy output to the executed joint offset."""
    scale = config.get('policy_action_scale', 1.0)
    clip = config['raw_action_clip']
    margin = config['joint_target_margin_rad']
    low, high = limits.unbind(-1)
    offset = (policy * scale + noise).clamp(-clip, clip)
    return (home + offset).clamp(low + margin, high - margin) - home


class EpisodeParameters:
    def __init__(self,env):
        self.env=env;device=env.device;n=env.num_envs
        m=env.sim.mj_model
        self.feet=[i for i in range(m.ngeom) if m.geom(i).name.endswith(('left_foot_collision','right_foot_collision'))]
        assert len(self.feet)==2
        self.trunk=next(i for i in range(m.nbody) if m.body(i).name.endswith('trunk_base'))
        self.nominal_friction=env.sim.get_default_field('geom_friction').clone()
        self.nominal_com=env.sim.get_default_field('body_ipos').clone()
        self.next_friction=torch.full((n,),float(self.nominal_friction[self.feet[0],0]),device=device)
        self.next_bias=torch.zeros((n,14),device=device);self.next_com=torch.zeros((n,3),device=device)
        self.friction=self.next_friction.clone();self.bias=self.next_bias.clone();self.com=self.next_com.clone()
    def queue(self,env_ids,*,friction,bias,com_offset):
        ids=torch.as_tensor(env_ids,dtype=torch.long,device=self.env.device)
        if ids.ndim!=1 or len(ids)==0 or torch.any(ids<0) or torch.any(ids>=self.env.num_envs) or ids.unique().numel()!=ids.numel():
            raise ValueError('env_ids must be unique valid world IDs')
        mu=torch.as_tensor(friction,dtype=torch.float32,device=self.env.device)
        b=torch.as_tensor(bias,dtype=torch.float32,device=self.env.device)
        c=torch.as_tensor(com_offset,dtype=torch.float32,device=self.env.device)
        if mu.shape!=(len(ids),) or b.shape!=(len(ids),14) or c.shape!=(len(ids),3):raise ValueError('Expected shapes [N], [N,14], [N,3]')
        for value,bounds in [(mu,SPEC['foot_sliding_friction_range']),(b,SPEC['joint_observation_bias_rad_range']),(c,SPEC['trunk_com_offset_m_range'])]:
            if not torch.isfinite(value).all() or torch.any(value<bounds[0]-1e-8) or torch.any(value>bounds[1]+1e-8):
                raise ValueError('Parameter outside configured initial range')
        # Queue only. Changing parameters mid-episode is deliberately unsupported.
        self.next_friction[ids]=mu;self.next_bias[ids]=b;self.next_com[ids]=c
    def apply_reset(self,env_ids):
        ids=torch.arange(self.env.num_envs,device=self.env.device) if env_ids is None else env_ids
        self.friction[ids]=self.next_friction[ids];self.bias[ids]=self.next_bias[ids];self.com[ids]=self.next_com[ids]
        friction=self.env.sim.model.geom_friction
        for foot in self.feet:
            # Rebuild from immutable compile-time default, preserving torsional/rolling coefficients.
            friction[ids,foot]=self.nominal_friction[foot]
            friction[ids,foot,0]=self.friction[ids]
        self.env.sim.model.body_ipos[ids,self.trunk]=self.nominal_com[self.trunk]+self.com[ids]
    def context(self):
        return torch.cat([self.friction[:,None],self.bias,self.com],dim=-1).clone()

@requires_model_fields('geom_friction','body_ipos',recompute=RecomputeLevel.set_const)
def reset_with_parameters(env,env_ids):
    reset_to_home(env,env_ids)
    if not hasattr(env,'episode_parameters'):env.episode_parameters=EpisodeParameters(env)
    env.episode_parameters.apply_reset(env_ids)

def configure_explicit_parameters(cfg):
    """Opt-in only; the existing reference recorder stays nominal unless called."""
    cfg.events['default_reset'].func=reset_with_parameters
    return cfg

def policy_observation(env,last_action,command,frame_noise=None):
    robot=env.scene['robot'];d=robot.data
    ids=[robot.joint_names.index(j['name']) for j in JOINTS]
    bl=[robot.joint_names.index('passive_'+j['name']+'_backlash') for j in JOINTS]
    position=d.joint_pos[:,ids]+d.joint_pos[:,bl]-d.default_joint_pos[:,ids]
    manager=getattr(env,'episode_parameters',None)
    if manager is not None:position=position+manager.bias
    if frame_noise is not None:
        if frame_noise.shape!=position.shape or not torch.isfinite(frame_noise).all():raise ValueError('Explicit finite noise [N,14] required')
        position=position+frame_noise
    return torch.cat([d.root_link_ang_vel_b,d.projected_gravity_b,position,
        d.joint_vel[:,ids]+d.joint_vel[:,bl],last_action,command],dim=-1)
