"""Runtime for the four-second right-foot kick."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensorCfg, ContactMatch
from microduck_sm.environment import TrainingEnvironment as BaseEnvironment, STALL_WINDOW
from microduck_sm.math_fk import DifferentiableFK, normalize, qconj, rotate, aligned_body_errors
from microduck_sm.old_aligned import OldAlignedPolicy, elu_mlp
from ball_rewards import lateral_ball_penalty
from ball_sampling import BallDifficultySampler, combined_difficulty_score, kick_attempt_eligible, qualified_kick_motion
from record_policy_reference import configuration
from environment_variations import configure_explicit_parameters, bounded_action, policy_observation
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG

ROOT = Path(__file__).resolve().parent


def ball_motion_metrics(delta,path_length):
    """Metres: signed forward displacement, lateral drift, XY displacement and travelled path."""
    return dict(
        mean_forward_distance_m=float(delta[:,0].mean()),
        max_forward_distance_m=float(delta[:,0].max()),
        mean_abs_lateral_distance_m=float(delta[:,1].abs().mean()),
        mean_planar_displacement_m=float(delta[:,:2].norm(dim=-1).mean()),
        mean_path_length_m=float(path_length.mean()),
    )


def ball_sensors():
    return tuple(ContactSensorCfg(name=f'{side}_ball',
        primary=ContactMatch(mode='geom',pattern=f'^{side}_foot_collision$',entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='^ball_geom$',entity='ball'),
        fields=('found',),reduce='netforce',num_slots=1) for side in ('left','right'))


class TrackingEnvironment(BaseEnvironment):
    def __init__(self,cfg):
        self.cfg=cfg;self.device=cfg.get('device','cuda:0');self.n=cfg['num_envs']
        ecfg=configure_explicit_parameters(configuration());ecfg.scene.num_envs=self.n
        ecfg.scene.entities['ball']=deepcopy(MICRODUCK_BALL_CFG)
        ecfg.scene.sensors=ball_sensors()
        ecfg.sim.nconmax=max(ecfg.sim.nconmax or 0,64)
        self.env=ManagerBasedRlEnv(ecfg,device=self.device)
        self.env.reset(seed=cfg['seed'])
        self.robot=self.env.scene['robot'];self.d=self.robot.data;self.a=self.robot.actuators[0]
        self.ids=torch.tensor([i for i,n in enumerate(self.robot.joint_names) if not n.startswith('passive_')],device=self.device)
        self.bl=torch.tensor([self.robot.joint_names.index('passive_'+self.robot.joint_names[i]+'_backlash') for i in self.ids.tolist()],device=self.device)
        assert len(self.ids)==14 and len(self.bl)==14
        data=np.load(ROOT/cfg['reference'])
        self.ref={k:torch.tensor(data[k],device=self.device) for k in data.files if np.issubdtype(data[k].dtype,np.number) or data[k].dtype==bool}
        self.reference_end=cfg.get('reference_end',650)
        if not 0<self.reference_end<len(self.ref['root_pose']):
            raise ValueError('Reference end is outside the recorded frames')
        self.speed_weights = None
        if cfg.get('speed_penalty'):
            reference_velocity = self.ref['root_velocity'][:self.reference_end + 1, 0].float()
            self.speed_weights = self.reference_speed_weights(reference_velocity)
        self.last=torch.zeros((self.n,14),device=self.device)
        self.frame=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.age=torch.zeros_like(self.frame);self.episode=torch.zeros_like(self.frame)
        self.history=torch.zeros((self.n,cfg['history_frames'],48),device=self.device)
        self.pose_history = None
        if cfg.get('stall_penalty'):
            pose_dim = 7 + len(self.robot.joint_names)
            self.pose_history = torch.zeros((self.n, STALL_WINDOW, pose_dim), device=self.device)
        self.next_episode_id=0
        self.fk=DifferentiableFK(self.env.sim.mj_model).to(self.device)
        self.offsets=torch.tensor(cfg['future_offsets'],device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(cfg['seed']+1)
        score_cfg=cfg['difficulty_score']
        if self.reference_end!=200 or score_cfg['bin_count']!=4 or cfg['reset_sampling']['hard_probability_cap']!=.4:
            raise ValueError('Expected 200 frames, four difficulty bins, and probability cap 0.4')
        self.reset_sampler=BallDifficultySampler(cfg['reset_sampling'],self.device,self.reference_end)
        self.schema=None
        self.sampling_logs=[[] for _ in range(self.n)]
        self.sampling_active=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        # mj_forward during a partial reset rebuilds efc friction forces globally.
        # Preserve the previous solver friction input until the next compute of
        # worlds that did not reset; the next physical substep supplies fresh forces.
        self._friction_override_mask=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self._friction_override=torch.zeros((self.n,self.env.sim.mj_model.nv),device=self.device)
        friction_read=self.a._dof_friction_force
        compute=self.a.compute
        def read_friction(nv):
            value=friction_read(nv)
            return torch.where(self._friction_override_mask[:,None],self._friction_override,value)
        def compute_preserving_history(command):
            torque=compute(command)
            self._friction_override_mask.zero_()
            return torque
        self.a._dof_friction_force=read_friction
        self.a.compute=compute_preserving_history
        self.ball=self.env.scene['ball']
        self.ball_start=torch.zeros(self.n,3,device=self.device)
        self.reset_calls=0
        self.kick_eligible=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.kick_checked=self.kick_eligible.clone();self.kick_failed=self.kick_eligible.clone()
        self.kick_right_touch=self.kick_eligible.clone();self.kick_left_touch=self.kick_eligible.clone()
    def state(self):
        first=self.schema is None
        state=super().state()
        pose=self.ball.data.root_link_pose_w.clone();pose[:,:3]-=self.env.scene.env_origins
        result=torch.cat((state,pose,self.ball.data.root_link_vel_w),-1)
        if first:self.schema['ball']=[state.shape[-1],result.shape[-1]]
        return result
    def reset(self,ids,frames=None):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        if not len(ids):return
        super().reset(ids,frames)
        keep=torch.ones(self.n,dtype=torch.bool,device=self.device);keep[ids]=False
        preserved={k:getattr(self.env.sim.data,k).clone() for k in ['qfrc_bias','qfrc_constraint','qfrc_actuator','qacc_warmstart']}
        pose=self.ref['ball_pose'][self.frame[ids],0].float().clone()
        pose[:,:3]+=self.env.scene.env_origins[ids]
        pose[:,:2]+=(torch.rand(len(ids),2,device=self.device,generator=self.generator)*2-1)*self.cfg['ball_position_noise']
        self.ball.write_root_link_pose_to_sim(pose,env_ids=ids)
        self.ball.write_root_link_velocity_to_sim(self.ref['ball_velocity'][self.frame[ids],0].float(),env_ids=ids)
        self.env.sim.forward();self.env.sim.sense()
        for k,v in preserved.items():getattr(self.env.sim.data,k)[keep]=v[keep]
        self.ball_start[ids]=pose[:,:3]
        spec=self.cfg['difficulty_score']
        speed=self.ball.data.root_link_lin_vel_w[ids].norm(dim=-1)
        eligible=kick_attempt_eligible(self.frame[ids],speed,spec['kick_end_frame'],spec['max_initial_ball_speed'])
        eligible&=~(self.contact('right')[ids]|self.contact('left')[ids])
        self.kick_eligible[ids]=eligible
        self.kick_checked[ids]=False;self.kick_failed[ids]=False
        self.kick_right_touch[ids]=False;self.kick_left_touch[ids]=False
        if frames is None:self.reset_sampler.record_initializations(eligible)
        self.reset_calls+=1
    def ball_tracking_reward(self,state,frame):
        a,b=self.schema['ball'];ball=state[:,a:b]
        target=self.ref['ball_pose'][frame,0].float()
        velocity=self.ref['ball_velocity'][frame,0].float()
        wp,sp=self.cfg['ball_reward']['position'];wv,sv=self.cfg['ball_reward']['velocity']
        return wp*torch.exp(-(ball[:,:3]-target[:,:3]).square().sum(-1)/sp**2)+wv*torch.exp(-(ball[:,7:10]-velocity[:,:3]).square().sum(-1)/sv**2)
    def ball_reward(self,state,frame):
        a,_=self.schema['ball']
        initial=self.ref['ball_pose'][0,0,:3].to(state)
        penalty=lateral_ball_penalty(state[:,a:a+3],initial,frame,self.cfg['ball_lateral_penalty'])
        return self.ball_tracking_reward(state,frame)-penalty
    def reward(self,action,previous,terminated):
        reward,terms=super().reward(action,previous,terminated)
        ball=self.ball_reward(self.state(),self.frame)
        return reward+ball,torch.cat((terms,self.ball_tracking_reward(self.state(),self.frame)[:,None]),-1)
    def step(self,action):
        # Reuse original physics, termination, history and reward. Replace only its
        # pose-only difficulty logging, with no change to replay batch sampling.
        active=self.sampling_active.clone();frames=self.frame.clone()
        self.sampling_active.zero_()
        try:result=super().step(action)
        finally:self.sampling_active.copy_(active)
        reward,terms,terminated,truncated,reason=result
        self.kick_right_touch|=self.contact('right')
        self.kick_left_touch|=self.contact('left')
        spec=self.cfg['difficulty_score']
        movement=self.ball.data.root_link_pos_w[:,0]-self.ball_start[:,0]
        checked=self.kick_eligible&~self.kick_checked&((self.frame>=spec['kick_end_frame'])|terminated|truncated)
        failed=checked&~self.kick_right_touch&(movement<spec['minimum_forward_movement_m'])
        qualified=qualified_kick_motion(self.kick_right_touch,self.kick_left_touch,movement,spec['minimum_forward_movement_m'])
        successful=checked&qualified
        self.kick_checked|=checked;self.kick_failed|=failed
        weights=[v[0] for v in self.cfg['reward'].values() if isinstance(v,list)]
        robot=terms[:,:len(weights)].sum(-1)/sum(weights)
        ball=terms[:,-1]/sum(v[0] for v in self.cfg['ball_reward'].values())
        unqualified=self.kick_eligible&self.kick_checked&~qualified
        score=combined_difficulty_score(robot,ball,unqualified,spec['robot_weight'],spec['ball_weight'])
        for i in active.nonzero().flatten().tolist():
            log=self.sampling_logs[i]
            log.append((frames[i].detach().clone(),score[i].detach().clone()))
            kick_frames=frames.new_empty(0)
            if failed[i] or successful[i]:
                selected=[f for f,_ in log if int(f)<spec['kick_end_frame']]
                if selected:kick_frames=torch.stack(selected)
            if failed[i]:
                self.sampling_logs[i]=log=[(f,torch.full_like(value,spec['failure_frame_score']) if int(f)<spec['kick_end_frame'] else value) for f,value in log]
                self.reset_sampler.mark_kick_failure(kick_frames)
            elif successful[i]:self.reset_sampler.mark_kick_success(kick_frames)
            if terminated[i] or truncated[i]:
                self.reset_sampler.update_episode(torch.stack([f for f,_ in log]),torch.stack([v for _,v in log]),terminated=bool(terminated[i]))
                self.sampling_logs[i]=[];self.sampling_active[i]=False
        return result
    def contact(self,side):
        return self.env.scene.sensors[f'{side}_ball'].data.found.reshape(self.n,-1).any(-1)


class TrackingPolicy(OldAlignedPolicy):
    def __init__(self,config):
        torch.nn.Module.__init__(self)
        self.target_encoder=elu_mlp(440,config['target_encoder_hidden'],32)
        self.decoder=elu_mlp(512,config['policy_hidden'],14,small_output=True)
        self.register_buffer('observation_scale',torch.tensor([.2]*3+[1.]*3+[1.]*14+[.1]*14+[1.]*14).repeat(10))
