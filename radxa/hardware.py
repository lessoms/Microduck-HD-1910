"""Calibrated Feetech HAT and Linux IIO body IMU, plus explicit mock hardware."""
import math, struct, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class Sample:
    stamp: float
    q: np.ndarray
    dq: np.ndarray
    gyro: np.ndarray
    gravity: np.ndarray
    current: dict

def vector(x,n):
    a=np.asarray(x,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Invalid sensor vector')
    return a

class MockHardware:
    simulated=True
    def __init__(self,home):self.q=np.array(home,dtype=float);self.target=self.q.copy();self.last=time.monotonic();self.closed=False
    def read(self):
        now=time.monotonic();dt=max(now-self.last,1e-3);old=self.q.copy();self.q+=np.clip(self.target-self.q,-.02,.02);self.last=now
        return Sample(now,self.q.copy(),(self.q-old)/dt,np.zeros(3),np.array([0.,0.,-1.]),{})
    def write(self,q,mouth=None):self.target=np.array(q,dtype=float)
    def close(self):self.closed=True

class IIOBodyIMU:
    def __init__(self,cfg):
        if not cfg.get('mount_verified'):raise ValueError('Body IMU mounting transform must be calibrated')
        self.cfg=cfg;self.rotation=np.asarray(cfg['sensor_to_body'],float)
        if self.rotation.shape!=(3,3) or not np.allclose(self.rotation@self.rotation.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(self.rotation),1,atol=1e-5):raise ValueError('Invalid sensor-to-body rotation')
        self.bias=vector(cfg['gyro_bias_rad_s'],3);self.gravity=None;self.last=None
    def channels(self,device,kind):
        d=Path(device);scale=float((d/f'in_{kind}_scale').read_text())
        values=[]
        for axis in 'xyz':
            p=d/f'in_{kind}_{axis}_offset';offset=float(p.read_text()) if p.exists() else 0.
            values.append((float((d/f'in_{kind}_{axis}_raw').read_text())+offset)*scale)
        return vector(values,3)
    def read(self):
        # Linux IIO accel uses m/s² and angular velocity rad/s; no degree conversion.
        acc=self.rotation@self.channels(self.cfg['accel_device'],'accel')
        gyro=self.rotation@self.channels(self.cfg['gyro_device'],'anglvel')-self.bias
        now=time.monotonic();norm=np.linalg.norm(acc)
        if norm<1e-6:raise ValueError('Invalid acceleration')
        measured=-acc/norm
        if self.gravity is None:
            if abs(norm-9.80665)>1.5:raise ValueError('Initialize IMU while stationary')
            self.gravity=measured
        else:
            dt=now-self.last
            if dt<=0 or dt>.1:raise ValueError('IMU sampling gap')
            pred=self.gravity-np.cross(gyro,self.gravity)*dt
            # Suppress accelerometer correction during substantial translational acceleration.
            alpha=1-math.exp(-dt/self.cfg.get('correction_tau_s',1.)) if abs(norm-9.80665)<1.5 else 0.
            self.gravity=(1-alpha)*pred+alpha*measured;self.gravity/=np.linalg.norm(self.gravity)
        self.last=now
        return gyro,self.gravity.copy()

class FeetechHardware:
    simulated=False
    def __init__(self,cfg,names,write_enabled=False):
        from feetech_bus import FeetechBus
        self.cfg=cfg;self.names=names;self.write_enabled=write_enabled;self.joints=cfg['joints']
        if set(self.joints)!=set(names):raise ValueError('Joint calibration must cover the exact 14 model joints')
        ids=[]
        for name in names:
            j=self.joints[name]
            if not j.get('verified') or j.get('id') is None or j.get('zero_tick') is None:raise ValueError('Uncalibrated joint: '+name)
            if j['direction'] not in (-1,1) or j['ticks_per_rad']<=0 or j['speed_rad_s_per_unit']<=0:raise ValueError('Invalid joint units: '+name)
            ids.append(j['id'])
        if len(set(ids))!=14 or any(not 0<=x<254 for x in ids):raise ValueError('Invalid or duplicate servo IDs')
        self.ids=ids;self.imu=IIOBodyIMU(cfg['body_imu'])
        self.mouth=cfg.get('mouth',{})
        if self.mouth.get('verified'):
            j=self.mouth
            if j['id'] in ids or j['direction'] not in (-1,1) or j['ticks_per_rad']<=0:raise ValueError('Invalid mouth calibration')
            if not 0<j.get('closing_trip_current_a',0)<j.get('max_current_a',0):raise ValueError('Mouth current thresholds must be calibrated')
        self.bus=FeetechBus(cfg['bus']['port'],cfg['bus']['baud'],timeout=.008)
    def read(self):
        states=self.bus.states(self.ids);q=[];dq=[];currents={}
        for name,sid in zip(self.names,self.ids):
            s=states[sid];j=self.joints[name]
            if s is None or s['err'] or s['status']:raise RuntimeError('Missing or alarm servo '+str(sid))
            q.append((s['pos']-j['zero_tick'])/j['ticks_per_rad']*j['direction'])
            dq.append(s['speed']*j['speed_rad_s_per_unit']*j['direction']);currents[name]=s['current_ma']/1000
        if self.mouth.get('verified'):
            s=self.bus.state(self.mouth['id'])
            if s is None or s['err'] or s['status']:raise RuntimeError('Missing or alarm mouth servo')
            currents['passive_mouth']=s['current_ma']/1000
        gyro,g=self.imu.read()
        return Sample(time.monotonic(),vector(q,14),vector(dq,14),gyro,g,currents)
    def write(self,q,mouth=None):
        if not self.write_enabled:return
        values=[]
        for name,angle in zip(self.names,vector(q,14)):
            j=self.joints[name]
            if not j['min_rad']<=angle<=j['max_rad']:raise ValueError('Hardware joint limit '+name)
            tick=round(j['zero_tick']+j['direction']*angle*j['ticks_per_rad'])
            if not 0<=tick<=32767:raise ValueError('Invalid target tick')
            values.append((j['id'],struct.pack('<H',tick)))
        if mouth is not None:
            j=self.mouth
            if not j.get('verified') or not j['min_rad']<=mouth<=j['max_rad']:raise ValueError('Mouth not calibrated')
            tick=round(j['zero_tick']+j['direction']*mouth*j['ticks_per_rad'])
            if not 0<=tick<=32767:raise ValueError('Invalid mouth target tick')
            values.append((j['id'],struct.pack('<H',tick)))
        self.bus.sync_write(42,2,values)
    def close(self):self.bus.close()
