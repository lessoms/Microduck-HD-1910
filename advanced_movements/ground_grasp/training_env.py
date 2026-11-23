"""Full-state grasp environment, teacher controller, reward and exact snapshot restore."""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from simulate import build_config,BODY_NAMES
from vector_mouth import VectorMouthCfg
from geometry import MOUTH_MIN_RAD,MOUTH_MAX_RAD,MOUTH_JOINT
from environment_variations import configure_explicit_parameters,policy_observation,bounded_action
from microduck_sm.environment import TrainingEnvironment as StateEncoder
from microduck_sm.math_fk import DifferentiableFK,normalize,qconj,rotate,quat6,qmul,aligned_body_errors

class GraspEnvironment:
 def __init__(self,spec,n=64,device='cuda:0'):
  self.spec=spec;self.cfg=deepcopy(spec['collection_config']);self.device=device;self.n=n;self.schema=None
  cfg=configure_explicit_parameters(build_config(.05));cfg.scene.num_envs=n
  robot=cfg.scene.entities['robot'];robot.articulation.actuators=(*robot.articulation.actuators[:-1],VectorMouthCfg(target_names_expr=(MOUTH_JOINT,)))
  self.env=ManagerBasedRlEnv(cfg,device=device);self.env.reset(seed=spec['config']['seed'])
  self.robot=self.env.scene['robot'];self.d=self.robot.data;self.a=self.robot.actuators[0];self.mouth=self.robot.actuators[-1]
  self.obj=self.env.scene['object'];self.ids=torch.tensor([self.robot.joint_names.index(x) for x in BODY_NAMES],device=device)
  self.bl=torch.tensor([self.robot.joint_names.index('passive_'+x+'_backlash') for x in BODY_NAMES],device=device)
  self.jaw=[self.robot.joint_names.index(MOUTH_JOINT)]
  self.home=self.d.default_joint_pos[0,self.ids].clone();self.limits=self.d.default_joint_pos_limits[0,self.ids].clone()
  self.last=torch.zeros(n,14,device=device);self.last_raw=self.last.clone()
  self.clock=torch.zeros(n,1,device=device);self.initial_z=torch.full((n,1),.025,device=device)
  self.peak=torch.zeros(n,1,device=device);self.upper=self.peak.clone();self.lower=self.peak.clone();self.success=self.peak.clone()
  self.history=torch.zeros(n,10,48,device=device)
  self.fk=DifferentiableFK(self.env.sim.mj_model).to(device)
  self.reference=None
  self.age=torch.zeros(n,dtype=torch.long,device=device)
  self.pose_history=torch.zeros(n,8,7+len(self.robot.joint_names),device=device)
  self._override_mask=torch.zeros(n,dtype=torch.bool,device=device)
  self._override=torch.zeros(n,self.env.sim.mj_model.nv,device=device)
  read=self.a._dof_friction_force;compute=self.a.compute
  def preserved(nv):return torch.where(self._override_mask[:,None],self._override,read(nv))
  def clear(cmd):
   out=compute(cmd);self._override_mask.zero_();return out
  self.a._dof_friction_force=preserved;self.a.compute=clear
  self.reset(torch.arange(n,device=device))
 def reset(self,ids,randomize=True):
  ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
  if not len(ids):return
  keep=torch.ones(self.n,dtype=torch.bool,device=self.device);keep[ids]=False
  data=self.env.sim.data
  fields=('qfrc_bias','qfrc_constraint','qfrc_actuator','qacc_warmstart')
  preserved={k:getattr(data,k)[:].clone() for k in fields}
  friction=self.a._dof_friction_force(self.env.sim.mj_model.nv).clone()
  count=len(ids)
  self.env.episode_parameters.queue(ids,friction=.5+.7*torch.rand(count,device=self.device) if randomize else torch.ones(count,device=self.device),
    bias=-.01+.02*torch.rand(count,14,device=self.device) if randomize else torch.zeros(count,14,device=self.device),
    com_offset=-.002+.004*torch.rand(count,3,device=self.device) if randomize else torch.zeros(count,3,device=self.device))
  self.env._reset_idx(ids)
  self.last[ids]=0.;self.last_raw[ids]=0.;self.clock[ids]=0.;self.initial_z[ids]=.025
  self.peak[ids]=0.;self.upper[ids]=0.;self.lower[ids]=0.;self.success[ids]=0.
  pose=self.obj.data.root_link_pose_w[ids].clone();pose[:,:3]=self.env.scene.env_origins[ids]+torch.tensor([.087,0,.025],device=self.device)
  if randomize:pose[:,:2]+=-.005+.01*torch.rand(count,2,device=self.device)
  self.obj.write_root_link_pose_to_sim(pose,env_ids=ids);self.obj.write_root_link_velocity_to_sim(torch.zeros(count,6,device=self.device),env_ids=ids)
  self.robot.set_joint_position_target(torch.full((count,1),MOUTH_MIN_RAD,device=self.device),joint_ids=self.jaw,env_ids=ids[:,None])
  self.env.sim.forward();self.env.sim.sense()
  for k,v in preserved.items():getattr(data,k)[keep]=v[keep]
  self._override[keep]=friction[keep];self._override_mask[keep]=True
  self.history[ids]=self.proprio()[ids,None].expand(-1,10,-1)
  self.age[ids]=0
  pose=self.d.root_link_pose_w.clone();pose[:,:3]-=self.env.scene.env_origins
  self.pose_history[ids]=torch.cat((pose[ids],self.d.joint_pos[ids]),-1)[:,None].expand(-1,8,-1)
 def proprio(self):return policy_observation(self.env,self.last,torch.zeros(self.n,13,device=self.device))[:,:48]
 def context(self):return self.env.episode_parameters.context()
 def state(self):
  holder=SimpleNamespace(a=self.a,d=self.d,env=self.env,last=self.last,schema=None)
  base=StateEncoder.state(holder);fields={k:base[:,a:b] for k,(a,b) in holder.schema.items()}
  # MuJoCo xpos/cvel are display caches from before integration. Freejoint
  # qpos/qvel are the actual next-step physical state; restoring cached xpos
  # loses one 5ms integration step and silently corrupts random-frame resets.
  def physical_root(entity):
   ix=entity.indexing
   pose=self.env.sim.data.qpos[:,ix.free_joint_q_adr].clone()
   vel=self.env.sim.data.qvel[:,ix.free_joint_v_adr].clone()
   worldvel=torch.cat((vel[:,:3],rotate(pose[:,3:7],vel[:,3:6])),-1)
   pose[:,:3]-=self.env.scene.env_origins
   return pose,worldvel
  fields['root_pose'],fields['root_velocity']=physical_root(self.robot)
  pose,objectvel=physical_root(self.obj)
  fields.update(object_pose=pose,object_velocity=objectvel,
   mouth_baseline=self.mouth.baseline,mouth_armed=self.mouth.armed.float(),mouth_latched=self.mouth.latched.float(),
   mouth_frozen=self.mouth.frozen,mouth_last_torque=self.mouth.last_torque,
   clock=self.clock,initial_object_z=self.initial_z,object_peak=self.peak,
   upper_contact_history=self.upper,lower_contact_history=self.lower,success=self.success)
  schema={};start=0
  for k,v in fields.items():schema[k]=[start,start+v.shape[-1]];start+=v.shape[-1]
  if self.schema is None:self.schema=schema
  assert schema==self.schema
  return torch.cat(list(fields.values()),-1).float().clone()
 def get(self,state,key):return state[...,slice(*self.schema[key])]
 def step(self,action):
  old=self.state();previous=self.last.clone()
  phase=self.clock/4;active=phase<.7
  target=(.386*((phase-.02)/.08).clamp(0,1)*(1-((phase-.34)/.06).clamp(0,1))).clamp(MOUTH_MIN_RAD,MOUTH_MAX_RAD)
  target=torch.where(active,target,MOUTH_MIN_RAD);target=torch.where(self.mouth.latched,self.mouth.frozen,target)
  self.mouth.armed=active&(phase>=.34)
  self.robot.set_joint_position_target(target,joint_ids=self.jaw)
  self.env.step(action);self.last=action.clone();self.clock+=.02
  self.upper=torch.maximum(self.upper,self.env.scene.sensors['object_upper_beak'].data.found.reshape(self.n,-1).any(-1,keepdim=True).float())
  self.lower=torch.maximum(self.lower,self.env.scene.sensors['object_lower_beak'].data.found.reshape(self.n,-1).any(-1,keepdim=True).float())
  lift=self.env.sim.data.qpos[:,self.obj.indexing.free_joint_q_adr[2:3]]-self.env.scene.env_origins[:,2:3]-self.initial_z
  self.peak=torch.maximum(self.peak,torch.where((self.upper*self.lower)>.5,lift.clamp_min(0),0.))
  self.success=torch.maximum(self.success,(self.peak>.16).float())
  self.history=torch.cat((self.history[:,1:],self.proprio()[:,None]),1)
  state=self.state();q=normalize(self.d.root_link_quat_w);up=1-2*(q[:,1]**2+q[:,2]**2)
  height=self.d.root_link_pos_w[:,2]-self.env.scene.env_origins[:,2]
  term=(up<math.cos(math.radians(55)))|(height<.07)|~torch.isfinite(state).all(-1)
  enc=self.d.joint_pos[:,self.ids]+self.d.joint_pos[:,self.bl]
  term|=(torch.maximum(self.limits[:,0]-enc,enc-self.limits[:,1]).amax(-1)>.1)
  done=term|(self.clock[:,0]>=4.999)
  self.age+=1
  reward=self.reward(old,state,action,term,done,self.pose_history[:,0],self.age)
  self.pose_history=torch.cat((self.pose_history[:,1:],torch.cat((state[:,:7],self.get(state,'joint_position')),-1)[:,None]),1)
  return state,reward,term,done
 def reward(self,old,state,action,term,done,past_pose=None,age=None):
  obj=self.spec['collection_config']['object_reward']
  inc=(self.get(state,'object_peak').clamp(0,.16)-self.get(old,'object_peak').clamp(0,.16)).clamp_min(0)[:,0]
  bonus=(self.get(state,'success')-self.get(old,'success')).clamp(0,1)[:,0]
  reward=obj['progress_total_reward']*inc/.16+obj['first_success_reward']*bonus-obj['episode_failure_penalty']*done.float()*(1-self.get(state,'success')[:,0])
  previous=self.get(old,'previous_action');r=self.cfg['reward']
  reward-=r['action_l2_penalty']*action.square().mean(-1)+r['action_delta_l2_penalty']*(action-previous).square().mean(-1)+r['fall_penalty']*term.float()
  if self.reference is not None:
   reward+=self.body_reward(state)
   if self.cfg.get('stall_penalty') and past_pose is not None:reward+=self.stall_cost(state,past_pose,age)
  return reward
 def pose_errors(self,root,joints,frame):
  ref=self.reference[frame];rp=self.get(ref,'joint_position');refroot=ref[:,:7]
  body,quat=self.fk(root,joints);rb,rq=self.fk(refroot,rp)
  rel=qmul(qconj(normalize(root[:,3:7])),normalize(refroot[:,3:7]))
  angle=2*torch.atan2(rel[:,1:].norm(dim=-1),rel[:,0].abs().clamp_min(1e-8))
  bp,bo=aligned_body_errors(root,refroot,body,quat,torch.cat((rb,rq),-1),True)
  return dict(root_position=(root[:,:3]-refroot[:,:3]).square().sum(-1),root_orientation=angle.square(),
   encoder_position=(joints[:,self.ids]+joints[:,self.bl]-rp[:,self.ids]-rp[:,self.bl]).square().mean(-1),body_position=bp,body_orientation=bo)
 def pose_distance(self,root,joints,frame):
  errors=self.pose_errors(root,joints,frame);weights=self.cfg['reward']
  return sum(-weights[k][0]*torch.expm1(-v/weights[k][1]**2) for k,v in errors.items())/sum(weights[k][0] for k in errors)
 def stall_cost(self,state,past_pose,age):
  frame=(self.get(state,'clock')[:,0]*50).round().long().clamp(0,250)
  cand=(frame[:,None]-torch.arange(5,device=self.device)).clamp_min(0)
  distance=self.pose_distance(state[:,:7].repeat_interleave(5,0),self.get(state,'joint_position').repeat_interleave(5,0),cand.flatten()).reshape(-1,5)
  best=distance.argmin(-1,keepdim=True);ref=cand.gather(1,best)[:,0];current=distance.gather(1,best)[:,0]
  past=self.pose_distance(past_pose[:,:7],past_pose[:,7:],ref)
  cost=-.5*((current-.20)/.25).clamp(0,1)*((.06-(past-current))/.045).clamp(0,1)
  return torch.where(age>=8,cost,0.)
 def body_reward(self,state):
  frame=(self.get(state,'clock')[:,0]*50).round().long().clamp(0,250);ref=self.reference[frame]
  qv=self.get(state,'joint_velocity');rv=self.get(ref,'joint_velocity')
  errors=self.pose_errors(state[:,:7],self.get(state,'joint_position'),frame)
  errors.update(root_linear_velocity=(state[:,7:10]-ref[:,7:10]).square().sum(-1),root_angular_velocity=(state[:,10:13]-ref[:,10:13]).square().sum(-1),
   encoder_velocity=(qv[:,self.ids]+qv[:,self.bl]-rv[:,self.ids]-rv[:,self.bl]).square().mean(-1))
  return sum(self.cfg['reward'][k][0]*torch.exp(-v.clamp_min(0)/self.cfg['reward'][k][1]**2) for k,v in errors.items())
 def future(self,frame):
  offsets=torch.tensor(self.cfg['future_offsets'],device=self.device);idx=(frame[:,None]+offsets).clamp(0,250)
  ref=self.reference[idx];base=self.reference[frame.clamp(0,250)];inverse=qconj(base[:,3:7])[:,None].expand(-1,10,-1)
  pose=self.get(ref,'root_pose');vel=self.get(ref,'root_velocity');qp=self.get(ref,'joint_position');qv=self.get(ref,'joint_velocity')
  return torch.cat((rotate(inverse,pose[:,:,:3]-base[:,None,:3]),quat6(qmul(inverse,normalize(pose[:,:,3:7]))),rotate(inverse,vel[:,:,:3]),rotate(inverse,vel[:,:,3:]),
    qp[:,:,self.ids]+qp[:,:,self.bl]-self.home,qv[:,:,self.ids]*.1+qv[:,:,self.bl]*.1,idx[:,:,None]/250),-1).flatten(1)
 def restore(self,state,context,ids):
  ids=torch.as_tensor(ids,device=self.device,dtype=torch.long);keep=torch.ones(self.n,device=self.device,dtype=torch.bool);keep[ids]=False
  data=self.env.sim.data;preserved={k:getattr(data,k)[:].clone() for k in ('qfrc_bias','qfrc_constraint','qfrc_actuator','qacc_warmstart')}
  friction=self.a._dof_friction_force(self.env.sim.mj_model.nv).clone()
  self.env.episode_parameters.queue(ids,friction=context[:,0],bias=context[:,1:15],com_offset=context[:,15:18]);self.env._reset_idx(ids)
  get=lambda k:self.get(state,k)
  pose=get('root_pose').clone();pose[:,:3]+=self.env.scene.env_origins[ids]
  self.robot.write_root_link_pose_to_sim(pose,env_ids=ids);self.robot.write_root_link_velocity_to_sim(get('root_velocity'),env_ids=ids)
  self.robot.write_joint_state_to_sim(get('joint_position'),get('joint_velocity'),env_ids=ids)
  op=get('object_pose').clone();op[:,:3]+=self.env.scene.env_origins[ids]
  self.obj.write_root_link_pose_to_sim(op,env_ids=ids);self.obj.write_root_link_velocity_to_sim(get('object_velocity'),env_ids=ids)
  a=self.a;a._bam_model.actuator.q_target_smooth[ids]=get('smoothed_target');a._prev_motor_torque[ids]=get('previous_motor_torque')
  a._goal_reset_pending[ids]=get('goal_reset_pending').bool()
  cb=a._delay_buffer._buffer
  queue=get('delay_queue').reshape(len(ids),5,14)
  start=(cb._pointer+1)%cb._max_len
  for k in range(cb._max_len):cb._buffer[(start+k)%cb._max_len,ids]=queue[:,k]
  cb._num_pushes[ids]=get('delay_valid_length').flatten().long()
  delay=a._delay_buffer;delay._current_lags[ids]=get('delay_lag').flatten().long();delay._step_count[ids]=get('delay_step_count').flatten().long();delay._phase_offsets[ids]=get('delay_phase').flatten().long()
  a._bam_model.actuator.vin[ids]=get('effective_voltage')
  for attr,key in (('baseline','mouth_baseline'),('armed','mouth_armed'),('latched','mouth_latched'),('frozen','mouth_frozen'),('last_torque','mouth_last_torque')):
   value=get(key);getattr(self.mouth,attr)[ids]=value.bool() if attr in ('armed','latched') else value
  for attr,key in (('last','previous_action'),('clock','clock'),('initial_z','initial_object_z'),('peak','object_peak'),('upper','upper_contact_history'),('lower','lower_contact_history'),('success','success')):getattr(self,attr)[ids]=get(key)
  self.robot.set_joint_position_target(self.home+self.last[ids],joint_ids=self.ids,env_ids=ids[:,None])
  self.robot.write_ctrl_to_sim(get('previous_motor_torque'),ctrl_ids=a.ctrl_ids,env_ids=ids)
  self.robot.write_ctrl_to_sim(get('mouth_last_torque'),ctrl_ids=self.mouth.ctrl_ids,env_ids=ids)
  self.env.sim.forward();self.env.sim.sense()
  for k,v in preserved.items():getattr(data,k)[keep]=v[keep]
  for key,field in (('previous_qfrc_bias','qfrc_bias'),('previous_qfrc_constraint','qfrc_constraint'),('previous_qfrc_actuator','qfrc_actuator')):getattr(data,field)[ids[:,None],a._dof_ids]=get(key)
  data.qacc_warmstart[ids]=get('solver_warmstart')
  data.qfrc_actuator[ids[:,None],self.robot.indexing.joint_v_adr[self.jaw]]=get('mouth_last_torque')
  self._override[ids[:,None],a._dof_ids]=get('previous_friction_force');self._override_mask[ids]=True
  self._override[keep]=friction[keep];self._override_mask[keep]=True
  self.env.sim.model.dof_frictionloss[ids[:,None],a._dof_ids]=get('friction_budget')
  self.history[ids]=self.proprio()[ids,None].expand(-1,10,-1)
  self.age[ids]=0
  pose=self.d.root_link_pose_w.clone();pose[:,:3]-=self.env.scene.env_origins
  self.pose_history[ids]=torch.cat((pose[ids],self.d.joint_pos[ids]),-1)[:,None].expand(-1,8,-1)
 def close(self):self.env.close()
