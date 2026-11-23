"""Meaningful software checks; no hardware writes or claims of physical validation."""
import json,time,unittest
from pathlib import Path
import numpy as np
from policy import Policies
from controller import Controller
from hardware import MockHardware,Sample,FeetechHardware
from runtime_math import gravity_from_quaternion,tilt_quaternion
R=Path(__file__).parent
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.p=Policies(Path(os.environ.get('MODEL_PATH',R/'combined_model.pt')),R/'assets')
        cls.cfg=json.loads((R/'config.json').read_text())
    def setup_action(self,action):
        skill={'sit':'sitstand','stand':'sitstand'}.get(action,action);frame={'sit':100,'stand':260,'roll':25}.get(action,0)
        q=self.p.refs[skill+'_q'][frame].copy();g=gravity_from_quaternion(self.p.refs[skill+'_pose'][frame,3:7])
        hw=MockHardware(q);c=Controller(self.p,hw,self.cfg);now=time.monotonic()
        for i in range(10):c.update(Sample(now+i*.02,q,np.zeros(14),np.zeros(3),g,{}),now+i*.02)
        return c,now+.2,q,g
    def packet(self,action):return {'action':action,'seq':1,'valid_for_ms':100,'vx':0.,'wz':0.}
    def test_all_separate_actions_and_no_history_reset(self):
        for action in ('sit','stand','roll','kick','grasp','flamingo','velocity'):
            with self.subTest(action=action):
                c,t,q,g=self.setup_action(action);before=np.array(c.history).copy()
                self.assertTrue(c.request(self.packet(action),t),c.note)
                self.assertEqual(len(c.history),10);np.testing.assert_allclose(np.array(c.history)[:,:6],before[:,:6])
                target,_=c.update(Sample(t,q,np.zeros(14),np.zeros(3),g,{}),t)
                self.assertTrue(np.isfinite(target).all());self.assertLessEqual(float(np.max(np.abs(target-q))),self.cfg['max_target_delta_rad']+1e-6)
    def test_expiry_and_one_shot_broker_watchdog(self):
        c,t,q,g=self.setup_action('velocity');p=self.packet('velocity');p['vx']=.1;self.assertTrue(c.request(p,t))
        c.update(Sample(t+.11,q,np.zeros(14),np.zeros(3),g,{}),t+.11);np.testing.assert_array_equal(c.command,[0,0,0])
        c,t,q,g=self.setup_action('kick');self.assertTrue(c.request(self.packet('kick'),t))
        c.request({'action':'stop','reason':'broker_watchdog'},t+.12);self.assertEqual(c.state,'executing')
    def test_no_reset_after_estop_or_explicit_stop(self):
        c,t,_,_=self.setup_action('kick');c.request(self.packet('kick'),t);c.request({'action':'stop'},t+.01)
        self.assertEqual(c.state,'latched_hold');self.assertFalse(c.request(self.packet('kick'),t+.02))
    def test_bad_entry_and_sensor_data(self):
        c,t,q,g=self.setup_action('kick');c.last_sample.q=q+1;self.assertFalse(c.request(self.packet('kick'),t))
        c,t,q,g=self.setup_action('kick')
        with self.assertRaises(ValueError):c.update(Sample(t-1,q,np.zeros(14),np.zeros(3),g,{}),t)
        with self.assertRaises(ValueError):c.update(Sample(t,q*float('nan'),np.zeros(14),np.zeros(3),g,{}),t)
    def test_default_real_hardware_refuses_uncalibrated_configuration(self):
        with self.assertRaises(ValueError):FeetechHardware(self.cfg['hardware'],self.p.meta['joint_names'])
    def test_gravity_round_trip(self):
        for g in ([0,0,-1],[.5,0,-np.sqrt(.75)],[0,1,0],[0,0,1]):
            np.testing.assert_allclose(gravity_from_quaternion(tilt_quaternion(g)),g,atol=1e-7)
    def test_landing_model_present(self):self.assertIsNotNone(self.p.landing)
if __name__=='__main__':unittest.main()
