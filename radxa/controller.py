"""Human-confirmed action requests, preserved history, local expiry and latched holds."""
import math,time
import numpy as np
from hardware import vector

class Controller:
    def __init__(self,policies,hardware,cfg):
        self.p=policies;self.hw=hardware;self.cfg=cfg;self.state='waiting_confirmation';self.skill='sitstand';self.frame=25
        self.history=[];self.previous=np.zeros(14,np.float32);self.target=None;self.last_sample=None;self.command=np.zeros(3,np.float32)
        self.velocity_expiry=0.;self.note='等待人工确认并请求动作';self.accepted_seq=None;self.mouth=None;self.mouth_latched=False;self.mouth_peak=0.
    def latch(self,reason):
        self.state='latched_hold';self.command[:]=0;self.note=reason
    def request(self,p,now=None):
        now=time.monotonic() if now is None else now;action=p.get('action')
        if action=='estop':self.latch('急停请求已锁存；保持最后目标，需本地重启解除');return False
        if action=='stop':
            # Broker's 100 ms lease applies to continuous velocity, not one-shot skills.
            if p.get('reason')=='broker_watchdog' and self.state!='velocity':return True
            if self.state=='velocity':self.command[:]=0;self.velocity_expiry=0.;self.note='速度目标归零；不宣称已停稳'
            elif self.state=='executing':self.latch('动作被停止，保持最后目标；需要人工处理')
            return True
        if self.state=='latched_hold':self.note='已锁存；需本地检查并重启';return False
        if action=='servo':self.note='部署入口暂不接受单关节控制，避免与策略争用';return False
        if action not in ('velocity','sit','stand','roll','kick','grasp','flamingo'):self.note='不支持的动作';return False
        if p.get('valid_for_ms')!=100:self.note='无效的命令期限';return False
        if action=='velocity':
            try:v=float(p['vx']);w=float(p['wz'])
            except (KeyError,ValueError,TypeError):self.note='无效速度';return False
            if not math.isfinite(v+w) or abs(v)>.15 or abs(w)>.75:self.note='速度超出策略范围 vx ±0.15 / wz ±0.75';return False
            if self.state=='velocity':
                self.command[:]=[v,0,w];self.velocity_expiry=now+.1;return True
        if self.state=='executing':self.note='当前动作尚未结束';return False
        if self.state=='velocity' and np.linalg.norm(self.command)>.001:self.note='先松开方向键，并人工确认停稳';return False
        if len(self.history)<10 or self.last_sample is None:self.note='等待十帧真实观测';return False
        skill={'sit':'sitstand','stand':'sitstand'}.get(action,action)
        frame={'sit':100,'stand':260,'roll':25}.get(action,0)
        qref=self.p.refs[skill+'_q'][frame];pose=self.p.refs[skill+'_pose'][frame]
        if now-self.last_sample.stamp>self.cfg['sensor_timeout_s']:self.note='反馈已过期';return False
        if np.max(np.abs(self.last_sample.q-qref))>self.cfg['entry_joint_error_rad']:self.note='起始关节姿态不符合动作要求';return False
        from runtime_math import gravity_from_quaternion
        expected=gravity_from_quaternion(pose[3:7])
        angle=math.acos(float(np.clip(np.dot(expected,self.last_sample.gravity),-1,1)))
        if angle>self.cfg['entry_tilt_error_rad'] or np.linalg.norm(self.last_sample.gyro)>self.cfg['entry_gyro_rad_s']:self.note='IMU 姿态或晃动不满足进入条件';return False
        if action=='grasp' and not self.hw.simulated and not self.cfg['hardware'].get('mouth',{}).get('verified'):
            self.note='夹取需先完成嘴部映射及电流保护标定';return False
        old_home=np.array(self.p.meta['skills'][self.skill]['home']);new_home=np.array(self.p.meta['skills'][skill]['home'])
        for row in self.history:
            row[6:20]+=old_home-new_home
            row[34:48]+=old_home-new_home
        self.skill=skill;self.frame=frame;self.action_name=action;self.accepted_seq=p.get('seq');self.start_frame=frame
        self.state='velocity' if action=='velocity' else 'executing';self.command[:]=[v,0,w] if action=='velocity' else [0,0,0]
        self.velocity_expiry=now+.1 if action=='velocity' else 0
        self.blend_from=self.target.copy();self.blend_step=0;self.mouth_latched=False
        self.mouth_peak=0.;self.note='已接受人工动作请求'
        return True
    def update(self,s,now=None):
        now=time.monotonic() if now is None else now
        for value,n in ((s.q,14),(s.dq,14),(s.gyro,3),(s.gravity,3)):vector(value,n)
        if abs(np.linalg.norm(s.gravity)-1)>.02 or now-s.stamp>self.cfg['sensor_timeout_s'] or s.stamp>now+.01:raise ValueError('Invalid or stale body IMU sample')
        if self.last_sample is not None and s.stamp<=self.last_sample.stamp:raise ValueError('Repeated or out-of-order sensor sample')
        self.last_sample=s
        if self.target is None:self.target=s.q.copy()
        home=np.array(self.p.meta['skills'][self.skill]['home'])
        # last_action is the realised offset in joint radians, not raw actor output.
        obs=np.r_[s.gyro,s.gravity,s.q-home,s.dq,self.target-home].astype(np.float32)
        self.history.append(obs);self.history=self.history[-10:]
        if self.state not in ('executing','velocity') or len(self.history)<10:return self.target,self.mouth
        m=self.p.meta['skills'][self.skill]
        # Stage-relative gravity check allows intentional bending and rolling.
        from runtime_math import gravity_from_quaternion
        pose=self.p.refs[self.skill+'_pose'][min(self.frame,m['end'])]
        expected=gravity_from_quaternion(pose[3:7]);error=math.acos(float(np.clip(np.dot(expected,s.gravity),-1,1)))
        if error>self.cfg['tracking_tilt_error_rad'] or np.linalg.norm(s.gyro)>self.cfg['max_gyro_rad_s']:
            self.latch('动作期间姿态超出参考容差');return self.target,self.mouth
        if self.state=='velocity' and now>self.velocity_expiry:self.command[:]=0;self.note='速度指令过期，目标归零'
        previous=self.target-home
        offset=self.p.action(self.skill,self.frame,np.array(self.history),self.command,previous)
        desired=home+offset
        self.blend_step+=1;weight=min(1.,self.blend_step/self.cfg['blend_ticks']);desired=self.blend_from+(desired-self.blend_from)*weight
        limit=self.cfg['max_target_delta_rad'];self.target+=np.clip(desired-self.target,-limit,limit)
        lim=np.array(m['limits']);self.target=np.clip(self.target,lim[:,0]+m['margin'],lim[:,1]-m['margin'])
        if self.skill=='grasp':
            phase=(self.frame*.02)/4
            jaw=float(np.clip(.386*np.clip((phase-.02)/.08,0,1)*(1-np.clip((phase-.34)/.06,0,1)),math.radians(5),math.radians(35))) if phase<.7 else math.radians(5)
            mouthcfg=self.cfg['hardware'].get('mouth',{})
            if not self.hw.simulated:
                current=s.current.get('passive_mouth')
                if current is None or not math.isfinite(current):self.latch('缺少嘴部电流反馈');return self.target,self.mouth
                if abs(current)>mouthcfg['max_current_a']:self.latch('嘴部电流超过标定上限');return self.target,self.mouth
                if .34<=phase<.7 and abs(current)>=mouthcfg['closing_trip_current_a']:self.mouth_latched=True
            if not self.mouth_latched:
                previousjaw=self.mouth if self.mouth is not None else math.radians(5)
                self.mouth=previousjaw+float(np.clip(jaw-previousjaw,-limit,limit))
        self.frame+=1
        if self.state=='executing':
            end=200 if self.action_name=='sit' else m['end']
            if self.frame>=end:self.state='waiting_confirmation';self.note='动作时序结束，保持末目标；请人工确认姿态，不代表任务成功'
        return self.target,self.mouth
