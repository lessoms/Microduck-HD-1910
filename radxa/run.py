#!/usr/bin/env python3
"""Radxa headless producer for the existing Control_Platform WebSocket broker."""
import argparse,asyncio,json,os,queue,time
from collections import deque
from pathlib import Path
import numpy as np
from hardware import MockHardware,FeetechHardware
from policy import Policies
from controller import Controller
from runtime_math import tilt_quaternion

async def run(args):
    cfg=json.loads(args.config.read_text());policies=Policies(args.model,args.assets)
    if args.enable_write and args.hardware!='feetech':raise ValueError('--enable-write only applies to feetech')
    if args.hardware=='feetech':
        if args.enable_write and not cfg.get('real_motion_verified'):raise ValueError('Fill and validate real_motion_verified calibration before enabling writes')
        hardware=FeetechHardware(cfg['hardware'],policies.meta['joint_names'],args.enable_write)
    else:hardware=MockHardware(policies.meta['skills']['sitstand']['home'])
    c=Controller(policies,hardware,cfg);pending=queue.SimpleQueue();seq=0;last=None;timing=deque(maxlen=10000);started=time.monotonic()
    async def cycle(ws=None):
        nonlocal seq,last
        deadline=time.monotonic()
        def step():
            nonlocal last
            now=time.monotonic()
            if last is not None and now-last>cfg['control_gap_s']:c.latch('控制周期超时')
            last=now
            while not pending.empty():
                packet,arrival=pending.get_nowait()
                if packet.get('action')!='estop' and now-arrival>.1:continue
                c.request(packet,now)
            try:
                sample=hardware.read();q,mouth=c.update(sample)
                # No target write at startup: require an explicitly accepted action.
                if c.accepted_seq is not None:hardware.write(q,mouth)
            except Exception as e:
                c.latch(str(e));raise
            return sample
        while not args.seconds or time.monotonic()-started<args.seconds:
            begin=time.monotonic()
            try:
                job=asyncio.create_task(asyncio.to_thread(step))
                try:sample=await asyncio.shield(job)
                except asyncio.CancelledError:
                    await job
                    raise
            except Exception as e:
                print(json.dumps({'fault':str(e),'state':c.state}),flush=True)
                if ws is not None:await ws.send_json({'type':'telemetry','seq':seq,'stamp_ms':time.monotonic()*1000,'joints_rad':{},'demo_only':hardware.simulated,'control':{'state':c.state,'message':c.note}})
                raise
            timing.append(time.monotonic()-begin);seq+=1
            if seq%2==0:
                message={'type':'telemetry','seq':seq,'stamp_ms':sample.stamp*1000,'joints_rad':dict(zip(policies.meta['joint_names'],sample.q.tolist())),
                         'body_wxyz':tilt_quaternion(sample.gravity),'demo_only':hardware.simulated,
                         'control':{'state':c.state,'skill':c.skill,'frame':c.frame,'message':c.note,'accepted_seq':c.accepted_seq,'write_enabled':bool(args.enable_write),'heading_observed':False}}
                if ws is not None:await ws.send_json(message)
            deadline+=.02;await asyncio.sleep(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline>.02:deadline=time.monotonic()
    try:
        if args.offline:await cycle()
        else:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(args.url,heartbeat=5) as ws:
                    await ws.send_json({'type':'hello','role':'producer','token':os.environ.get('CONTROL_TOKEN','')})
                    greeting=await ws.receive_json(timeout=5)
                    if greeting.get('type')!='welcome':raise RuntimeError('Broker rejected producer')
                    async def receive():
                        owner=None
                        async for msg in ws:
                            if msg.type!=aiohttp.WSMsgType.TEXT:continue
                            p=json.loads(msg.data)
                            if p.get('type')=='command':
                                owner=p.get('from',owner);pending.put((p,time.monotonic()))
                            elif p.get('type')=='peer_left' and p.get('id')==owner:
                                pending.put(({'action':'estop'},time.monotonic()))
                        c.latch('网络断开，需人工处理')
                    tasks=[asyncio.create_task(cycle(ws)),asyncio.create_task(receive())]
                    try:
                        done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                        for t in done:t.result()
                    finally:
                        for t in tasks:t.cancel()
                        await asyncio.gather(*tasks,return_exceptions=True)
    finally:
        c.latch('控制程序结束')
        # Wait for any in-flight worker before closing the bus.
        await asyncio.to_thread(lambda:None)
        hardware.close()
        if timing:print(json.dumps({'ticks':seq,'step_p50_ms':float(np.percentile(list(timing),50)*1000),'step_p99_ms':float(np.percentile(list(timing),99)*1000),'simulated':hardware.simulated}),flush=True)

def main():
    root=Path(__file__).resolve().parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=root/'combined_model.pt');p.add_argument('--assets',type=Path,default=root/'assets');p.add_argument('--config',type=Path,default=root/'config.json')
    p.add_argument('--url',default='ws://127.0.0.1:18080/ws');p.add_argument('--hardware',choices=('mock','feetech'),default='mock')
    p.add_argument('--enable-write',action='store_true');p.add_argument('--offline',action='store_true');p.add_argument('--seconds',type=float,default=0)
    a=p.parse_args()
    if a.seconds<0:p.error('--seconds must be nonnegative')
    if a.offline and not a.seconds:p.error('--offline requires a finite --seconds')
    asyncio.run(run(a))
if __name__=='__main__':main()
