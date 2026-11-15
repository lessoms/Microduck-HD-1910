import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'runtime'))
from robot_config import configuration
from environment_variations import bounded_action,configure_explicit_parameters,policy_observation
from mjlab.envs import ManagerBasedRlEnv
from .math_fk import DifferentiableFK,qmul,qconj,rotate,quat6,normalize,aligned_body_errors
from .reset_sampling import ResetFrameSampler

STALL_WINDOW = 8
STALL_REFERENCE_LAG = 4
SPEED_TOLERANCE = (.02, .20)
SPEED_PENALTY_WIDTH = (.04, .40)


class TrainingEnvironment:
    def __init__(self,cfg):
        self.cfg=cfg;self.device='cuda:0';self.n=cfg['num_envs']
        ecfg=configure_explicit_parameters(configuration());ecfg.scene.num_envs=self.n
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
        self.reset_sampler=ResetFrameSampler(cfg.get('reset_sampling'),self.device,self.reference_end)
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
    def get(self,key,frame=None):
        value=self.ref[key][self.frame if frame is None else frame]
        return value[:,0] if value.ndim>1 and value.shape[1]==1 else value
    def proprio(self):
        return policy_observation(self.env,self.last,torch.zeros((self.n,13),device=self.device))[:,:48]
    def _restore_ring(self,ids,queue):
        cb=self.a._delay_buffer._buffer
        start=(cb._pointer+1)%cb._max_len
        for k in range(cb._max_len):cb._buffer[(start+k)%cb._max_len,ids]=queue[:,k]
        cb._num_pushes[ids]=cb._max_len
    def reset(self,ids,frames=None):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        if not len(ids):return
        keep=torch.ones(self.n,dtype=torch.bool,device=self.device);keep[ids]=False
        # Use manager reset without the upstream global actuator compute, which would
        # advance every world's delay queue even when only one world resets.
        friction_history=self.a._dof_friction_force(self.env.sim.mj_model.nv).clone()
        preserved={k:getattr(self.env.sim.data,k)[:].clone() for k in ['qfrc_bias','qfrc_constraint','qfrc_actuator','qacc_warmstart']}
        sampling_reset=frames is None
        frames=self.reset_sampler.draw(len(ids),self.generator) if frames is None else torch.as_tensor(frames,dtype=torch.long,device=self.device)
        self.sampling_active[ids]=sampling_reset
        for i in ids.tolist():self.sampling_logs[i]=[]
        assert frames.shape==(len(ids),) and ((frames>=0)&(frames<self.reference_end)).all()
        mu=.5+.7*torch.rand(len(ids),device=self.device,generator=self.generator)
        bias=-.01+.02*torch.rand((len(ids),14),device=self.device,generator=self.generator)
        com=-.002+.004*torch.rand((len(ids),3),device=self.device,generator=self.generator)
        self.env.episode_parameters.queue(ids,friction=mu,bias=bias,com_offset=com)
        self.env._reset_idx(ids)
        self.frame[ids]=frames;self.age[ids]=0
        self.episode[ids]=torch.arange(self.next_episode_id,self.next_episode_id+len(ids),device=self.device)
        self.next_episode_id+=len(ids)
        pose=self.ref['root_pose'][frames,0].clone();pose[:,:3]+=self.env.scene.env_origins[ids]
        self.robot.write_root_link_pose_to_sim(pose,env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(self.ref['root_velocity'][frames,0].float(),env_ids=ids)
        self.robot.write_joint_state_to_sim(self.ref['qpos'][frames,0,7:].float(),self.ref['qvel'][frames,0,6:].float(),env_ids=ids)
        targets=self.ref['joint_target'][frames,0].float()
        self.robot.set_joint_position_target(targets,joint_ids=self.ids,env_ids=ids[:,None])
        self.a._bam_model.actuator.q_target_smooth[ids]=self.ref['actuator_smooth'][frames,0].float()
        self.a._prev_motor_torque[ids]=self.ref['previous_motor_torque'][frames,0].float()
        self.a._goal_reset_pending[ids]=False
        self._restore_ring(ids,self.ref['delay_queue'][frames,0].float())
        delay=self.a._delay_buffer
        delay._current_lags[ids]=self.ref['delay_lag'][frames,0].long()
        delay._step_count[ids]=self.ref['delay_step_count'][frames,0].long()
        delay._phase_offsets[ids]=self.ref['delay_phase'][frames,0].long()
        self.a._bam_model.actuator.vin[ids]=self.ref['voltage'][frames,0].float()
        self.robot.write_ctrl_to_sim(self.a._prev_motor_torque[ids],ctrl_ids=self.a.ctrl_ids,env_ids=ids)
        self.last[ids]=self.ref['observation'][frames,0,34:48].float()
        self.env.sim.forward();self.env.sim.sense()
        # Preserve historical solver-force inputs in worlds that did not reset.
        for k,v in preserved.items():getattr(self.env.sim.data,k)[keep]=v[keep]
        self._friction_override[keep]=friction_history[keep]
        self._friction_override_mask[keep]=True
        self._friction_override_mask[ids]=False
        self.history[ids]=self.proprio()[ids,None,:].expand(-1,self.cfg['history_frames'],-1)
        if self.pose_history is not None:
            reset_pose = torch.cat((pose[:, :3] - self.env.scene.env_origins[ids],
                                    pose[:, 3:], self.d.joint_pos[ids]), dim=-1)
            self.pose_history[ids] = reset_pose[:, None, :]
    def state(self):
        a=self.a;d=self.d
        pose=d.root_link_pose_w.clone();pose[:,:3]-=self.env.scene.env_origins
        data=self.env.sim.data
        fields={
            'root_pose':pose,'root_velocity':d.root_link_vel_w,
            'joint_position':d.joint_pos,'joint_velocity':d.joint_vel,
            'smoothed_target':a._bam_model.actuator.q_target_smooth,'previous_motor_torque':a._prev_motor_torque,
            'delay_queue':a._delay_buffer._buffer.buffer.flatten(1),
            'delay_lag':a._delay_buffer.current_lags[:,None].float(),
            'delay_step_count':a._delay_buffer._step_count[:,None].float(),
            'delay_phase':a._delay_buffer._phase_offsets[:,None].float(),
            'delay_valid_length':a._delay_buffer._buffer.current_length[:,None].float(),
            'goal_reset_pending':a._goal_reset_pending.float(),
            'previous_qfrc_bias':data.qfrc_bias[:,a._dof_ids],
            'previous_qfrc_constraint':data.qfrc_constraint[:,a._dof_ids],
            'previous_qfrc_actuator':data.qfrc_actuator[:,a._dof_ids],
            'previous_friction_force':a._dof_friction_force(self.env.sim.mj_model.nv)[:,a._dof_ids],
            'solver_warmstart':data.qacc_warmstart[:],
            'friction_budget':self.env.sim.model.dof_frictionloss[:,a._dof_ids],
            'effective_voltage':a._bam_model.actuator.vin,'previous_action':self.last}
        if self.schema is None:
            self.schema={};start=0
            for k,v in fields.items():self.schema[k]=[start,start+v.shape[1]];start+=v.shape[1]
        return torch.cat([v.float() for v in fields.values()],-1).clone()
    def future(self, frame=None):
        frame = self.frame if frame is None else frame
        indices=(frame[:,None]+self.offsets).clamp(max=self.reference_end)
        refroot=self.ref['root_pose'][indices,0].float()
        base=self.get('root_pose',frame).float();qbase=qconj(base[:,3:7])[:,None].expand(-1,len(self.offsets),-1)
        relpos=rotate(qbase,refroot[:,:,:3]-base[:,None,:3])
        orientation=quat6(qmul(qbase,refroot[:,:,3:7]))
        velocity=self.ref['root_velocity'][indices,0].float()
        linear=rotate(qbase,velocity[:,:,:3]);angular=rotate(qbase,velocity[:,:,3:])
        qp=self.ref['encoder_position'][indices,0].float()-self.d.default_joint_pos[:1,None,self.ids]
        qv=self.ref['encoder_velocity'][indices,0].float()*.1
        phase=(indices.clamp(max=self.cfg.get('reference_phase_cap',650)).float()/650)[:,:,None]
        return torch.cat([relpos,orientation,linear,angular,qp,qv,phase],-1).flatten(1)
    def executed_action(self,policy,noise):
        home = self.d.default_joint_pos[:, self.ids]
        limits = self.d.default_joint_pos_limits[:, self.ids]
        return bounded_action(policy, noise, home, limits, self.cfg)
    def step(self,action):
        sampling_frames=self.frame.clone()
        previous=self.last.clone();self.env.step(action)
        self.last=action.clone();self.age+=1;self.frame=(self.frame+1).clamp(max=self.reference_end)
        self.history=torch.cat([self.history[:,1:],self.proprio()[:,None]],1)
        pose=self.d.root_link_pose_w.clone();pose[:,:3]-=self.env.scene.env_origins
        tilt=torch.rad2deg(torch.acos((1-2*(pose[:,4]**2+pose[:,5]**2)).clamp(-1,1)))
        q=self.d.joint_pos[:,self.ids]+self.d.joint_pos[:,self.bl]
        bounds=self.d.default_joint_pos_limits[:,self.ids]
        violation=torch.maximum(bounds[:,:,0]-q,q-bounds[:,:,1]).clamp_min(0).amax(-1)
        finite=torch.isfinite(self.d.joint_pos).all(-1)&torch.isfinite(pose).all(-1)
        error=(pose[:,:3]-self.get('root_pose')[:,:3]).norm(dim=-1)
        term=self.cfg['termination'];reason=torch.zeros(self.n,dtype=torch.int32,device=self.device)
        height = pose[:, 2]
        violation_limit = term['maximum_encoder_limit_violation_rad']
        if self.cfg.get('relative_termination'):
            reference = self.get('root_pose')
            relative = qmul(qconj(normalize(reference[:, 3:])), normalize(pose[:, 3:]))
            tilt = torch.rad2deg(2 * torch.atan2(
                relative[:, 1:].norm(dim=-1), relative[:, 0].abs()
            ))
            height = height - reference[:, 2] + self.ref['root_pose'][0, 0, 2]
            reference_q = self.get('encoder_position')
            reference_violation = torch.maximum(
                bounds[:, :, 0] - reference_q, reference_q - bounds[:, :, 1]
            ).clamp_min(0).amax(-1)
            violation_limit = (reference_violation + 0.01).clamp_min(violation_limit)
        for bit,mask in [(1,tilt>term['max_tilt_deg']),(2,height<term['minimum_height_m']),
            (4,violation>violation_limit),(8,~finite),
            (16,error>term['maximum_reference_position_error_m'])]:reason|=mask.int()*bit
        terminated=reason!=0
        reason|=(self.frame>=self.reference_end).int()*32
        reason|=(self.age>=self.cfg['max_episode_steps']).int()*64
        truncated=((reason&96)!=0)&~terminated
        reward,terms=self.reward(action,previous,terminated)
        if self.pose_history is not None:
            current_pose = torch.cat((pose, self.d.joint_pos), dim=-1)
            self.pose_history = torch.cat((self.pose_history[:, 1:], current_pose[:, None]), dim=1)
        if self.cfg.get('reset_sampling',{}).get('mode') in ('adaptive_return','adaptive_window'):
            tracking_weights = [v[0] for v in self.cfg['reward'].values() if isinstance(v, list)]
            score = terms[:, :len(tracking_weights)].sum(-1) / sum(tracking_weights)
            if self.cfg.get('reset_speed_weight', 0.) and self.speed_weights is not None:
                reference, _ = self.closest_pose_reference(
                    pose, q, *self.fk(pose, self.d.joint_pos), self.frame
                )
                speed_cost = -self.speed_penalty(self.d.root_link_vel_w, self.frame, reference)
                # Full combined speed cost is 0.30. Keep the difficulty score in [0, 1].
                score = (score - self.cfg['reset_speed_weight'] * speed_cost / .30).clamp(0, 1)
            sampler_value=score if self.cfg['reset_sampling']['mode']=='adaptive_window' else reward
            for i in self.sampling_active.nonzero().flatten().tolist():
                self.sampling_logs[i].append((sampling_frames[i].detach(),sampler_value[i].detach()))
                if terminated[i] or truncated[i]:
                    log=self.sampling_logs[i]
                    # Reference completion is a finite task end. Exclude censoring at time limits.
                    if self.cfg['reset_sampling']['mode']=='adaptive_window' or terminated[i] or self.frame[i]>=self.reference_end:
                        self.reset_sampler.update_episode(torch.stack([x[0] for x in log]),torch.stack([x[1] for x in log]),terminated=bool(terminated[i]))
                    self.sampling_logs[i]=[]
                    self.sampling_active[i]=False
        torch.testing.assert_close(self.d.joint_pos_target[:,self.ids],self.d.default_joint_pos[:,self.ids]+action)
        return reward,terms,terminated,truncated,reason
    def reward(self,action,previous,terminated):
        root=self.d.root_link_pose_w.clone();root[:,:3]-=self.env.scene.env_origins
        enc=self.d.joint_pos[:,self.ids]+self.d.joint_pos[:,self.bl]
        encv=self.d.joint_vel[:,self.ids]+self.d.joint_vel[:,self.bl]
        past_pose = None
        if self.pose_history is not None:
            past_pose = self.pose_history[:, 0]
        return self.tracking_reward(
            root, self.d.root_link_vel_w, self.d.joint_pos, enc, encv,
            action, previous, terminated, self.frame, past_pose, self.age >= STALL_WINDOW,
        )

    def pose_errors(self, root, enc, body, body_quat, frame):
        refroot = self.get('root_pose', frame).float()
        relative = qmul(qconj(normalize(root[:, 3:])), normalize(refroot[:, 3:]))
        angle = 2 * torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs().clamp_min(1e-8))
        bp, bo = aligned_body_errors(
            root, refroot, body, body_quat, self.get('body_pose', frame).float(),
            self.cfg.get('body_reference_alignment') == 'robot_xy_yaw',
        )
        errors = {
            'root_position': (root[:, :3] - refroot[:, :3]).square().sum(-1),
            'root_orientation': angle.square(),
            'encoder_position': (enc - self.get('encoder_position', frame)).square().mean(-1),
            'body_position': bp,
        }
        if 'body_orientation' in self.cfg['reward']:
            errors['body_orientation'] = bo
        return errors

    def pose_distance(self, root, enc, body, body_quat, frame):
        errors = self.pose_errors(root, enc, body, body_quat, frame)
        weights = self.cfg['reward']
        gaps = [-weights[key][0] * torch.expm1(-error / weights[key][1]**2)
                for key, error in errors.items()]
        return torch.stack(gaps, dim=-1).sum(-1) / sum(weights[key][0] for key in errors)

    def closest_pose_reference(self, root, enc, body, body_quat, frame):
        candidates = (frame[:, None] - torch.arange(STALL_REFERENCE_LAG + 1, device=frame.device)).clamp_min(0)
        count = candidates.shape[1]
        distances = self.pose_distance(
            root.repeat_interleave(count, 0), enc.repeat_interleave(count, 0),
            body.repeat_interleave(count, 0), body_quat.repeat_interleave(count, 0),
            candidates.flatten(),
        ).reshape_as(candidates)
        closest = distances.argmin(dim=1, keepdim=True)
        return candidates.gather(1, closest).squeeze(1), distances.gather(1, closest).squeeze(1)

    def stall_penalty(self, reference, distance, past_pose, valid):
        past_root, past_joints = past_pose[:, :7], past_pose[:, 7:]
        past_enc = past_joints[:, self.ids] + past_joints[:, self.bl]
        past_body, past_quat = self.fk(past_root, past_joints)
        progress = self.pose_distance(past_root, past_enc, past_body, past_quat, reference) - distance
        gap_weight = ((distance - .20) / .25).clamp(0, 1)
        stall_weight = ((.06 - progress) / .045).clamp(0, 1)
        return torch.where(valid, -.5 * gap_weight * stall_weight, 0.)

    @staticmethod
    def reference_speed_weights(velocity):
        squared_speed = velocity.reshape(-1, 2, 3).square().sum(-1)
        # Center the 11-frame window on the current reference, repeating end frames.
        padded = F.pad(squared_speed.T[None], (5, 5), mode='replicate')
        rms = F.avg_pool1d(padded, 11, stride=1)[0].T.sqrt()
        activity = (rms / velocity.new_tensor(SPEED_TOLERANCE)).amax(dim=-1)
        return .25 + .75 * ((3 - activity) / 2).clamp(0, 1)

    def speed_penalty(self, velocity, frame, reference):
        target = self.get('root_velocity', reference).float().reshape(-1, 2, 3)
        error = (velocity.reshape(-1, 2, 3) - target).norm(dim=-1)
        tolerance = velocity.new_tensor(SPEED_TOLERANCE) + .2 * target.norm(dim=-1)
        excess = ((error - tolerance) / velocity.new_tensor(SPEED_PENALTY_WIDTH)).clamp(0, 1)
        return -.15 * excess.sum(dim=-1) * self.speed_weights[frame]

    def termination_probability(self, state, frame, softness=.05):
        """Smooth HD-1910 termination rules, relative to the rolling reference."""
        root = state[:, :7]
        joints = state[:, 13:41]
        reference = self.get('root_pose', frame).float()
        term = self.cfg['termination']
        relative = qmul(qconj(normalize(reference[:, 3:])), normalize(root[:, 3:]))
        # Clamp the acos argument to keep gradients finite at zero/180-degree error.
        cosine = (2 * relative[:, :1].square() - 1).squeeze(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        tilt = torch.rad2deg(torch.acos(cosine))
        height = root[:, 2] - reference[:, 2] + self.ref['root_pose'][0, 0, 2]
        enc = joints[:, self.ids] + joints[:, self.bl]
        bounds = self.d.default_joint_pos_limits[0, self.ids]
        violation = torch.maximum(bounds[:, 0] - enc, enc - bounds[:, 1]).clamp_min(0).amax(-1)
        reference_q = self.get('encoder_position', frame).float()
        reference_violation = torch.maximum(
            bounds[:, 0] - reference_q, reference_q - bounds[:, 1]
        ).clamp_min(0).amax(-1)
        limit = (reference_violation + .01).clamp_min(term['maximum_encoder_limit_violation_rad'])
        position_error = (root[:, :3] - reference[:, :3]).norm(dim=-1)
        # Residuals are dimensionless, zero on each corresponding real boundary.
        residual = torch.stack((
            tilt / term['max_tilt_deg'] - 1,
            (term['minimum_height_m'] - height) / .03,
            (violation - limit) / .05,
            position_error / term['maximum_reference_position_error_m'] - 1,
        ), -1).amax(-1)
        probability = torch.sigmoid(residual / softness)
        finite = torch.isfinite(state).all(-1)
        return torch.where(finite, probability, torch.ones_like(probability))

    def tracking_reward(self,root,vel,joints,enc,encv,action,previous,terminated,frame,
                        past_pose=None,stall_valid=None):
        refvel=self.get('root_velocity',frame).float()
        body,bq=self.fk(root,joints)
        errors=self.pose_errors(root,enc,body,bq,frame)
        # Keep the original tracking component order for metrics and reset sampling.
        errors={'root_position':errors['root_position'],
            'root_orientation':errors['root_orientation'],
            'root_linear_velocity':(vel[:,:3]-refvel[:,:3]).square().sum(-1),
            'root_angular_velocity':(vel[:,3:]-refvel[:,3:]).square().sum(-1),
            'encoder_position':errors['encoder_position'],
            'encoder_velocity':(encv-self.get('encoder_velocity',frame)).square().mean(-1),
            'body_position':errors['body_position'],
            **({'body_orientation': errors['body_orientation']} if 'body_orientation' in errors else {})}
        components=[self.cfg['reward'][k][0]*torch.exp(-err/self.cfg['reward'][k][1]**2) for k,err in errors.items()]
        components+=[-self.cfg['reward']['action_l2_penalty']*action.square().mean(-1),
            -self.cfg['reward']['action_delta_l2_penalty']*(action-previous).square().mean(-1),
            -self.cfg['reward']['fall_penalty']*terminated.float()]
        if past_pose is not None or self.speed_weights is not None:
            reference, distance = self.closest_pose_reference(root, enc, body, bq, frame)
            if past_pose is not None:
                components.append(self.stall_penalty(reference, distance, past_pose, stall_valid))
            if self.speed_weights is not None:
                components.append(self.speed_penalty(vel, frame, reference))
        terms=torch.stack(components,-1)
        return terms.sum(-1),terms
    def close(self):self.env.close()
