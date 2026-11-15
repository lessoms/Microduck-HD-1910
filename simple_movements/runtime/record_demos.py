"""Record real policy-driven MuJoCo rollouts and render README GIF/MP4 clips."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from button_skills import ButtonController, DEFAULT_SIT, DEFAULT_ROLL

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    'walking': 'Forward walking',
    'turning': 'Turning left',
    'backward': 'Backward walking',
    'sitstand': 'Sit down and stand up',
    'roulade': 'Forward roll and recovery',
}


def record(name, seed):
    c = ButtonController(DEFAULT_SIT, DEFAULT_ROLL, seed)
    frames, states = [], []
    start_tick = 0
    def step(capture=False):
        c.step()
        if c.state == 'fault':
            raise RuntimeError(f'{name} fault: {c.events[-1]}')
        if capture and (c.ticks - start_tick) % 2 == 0:
            frames.append(c.env.env.sim.data.qpos[0].cpu().numpy().copy())
            states.append(dict(t=round((c.ticks-start_tick)*.02, 3),state=c.state,
                               command=c.applied_velocity[0].cpu().tolist(),
                               root_velocity=c.env.d.root_link_vel_w[0].cpu().tolist()))
    def until_ready(state, capture=False, limit=900):
        for _ in range(limit):
            step(capture)
            if c.state == state and c.stable_ticks >= 8 and c.idle_ticks >= 35:
                return
        raise RuntimeError(f'{name}: timeout waiting for {state}, got {c.state}')
    try:
        until_ready('standing')
        if name in ('walking','turning','backward'):
            vx,wz={'walking':(.18,0.),'turning':(0.,.75),'backward':(-.18,0.)}[name]
            assert c.set_velocity(vx,wz)
            # Complete the original zero-command entry preparation before clipping.
            while c.state != 'walking':step()
            start_tick=c.ticks
            for _ in range(250):step(True)
            assert c.set_velocity(0.,0.)
            for _ in range(100):step(True)
            until_ready('standing')
        else:
            start_tick=c.ticks
            for _ in range(20):step(True)
            if name=='sitstand':
                assert c.command('sit')
                until_ready('sitting',True)
                for _ in range(25):step(True)
                assert c.command('stand')
                until_ready('standing',True)
            else:
                assert c.command('roll')
                until_ready('standing',True)
            for _ in range(30):step(True)
        assert c.reset_count == 1
        model=copy.copy(c.env.env.sim.mj_model)
        positions=np.asarray(frames)[:,:3]
        quats=np.asarray(frames)[:,3:7]
        yaw=np.unwrap(np.arctan2(2*(quats[:,0]*quats[:,3]+quats[:,1]*quats[:,2]),1-2*(quats[:,2]**2+quats[:,3]**2)))
        report=dict(planar_displacement_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),
                    skill=name,seed=seed,source='MuJoCo physics rollout of packaged trained actors',
                    fps=25,frames=len(frames),duration_s=len(frames)/25,
                    final_state=c.state,reset_count=c.reset_count,events=c.events,
                    state_samples=states,model_manifest_sha256=hashlib.sha256((ROOT/'models/manifest.json').read_bytes()).hexdigest())
        return model,np.asarray(frames),report
    finally:
        c.close()


def render(name,model,frames,report,width=640,height=480):
    media=ROOT/'media';gifs=media/'gifs';videos=media/'videos'
    for p in (gifs,videos,ROOT/'outputs/demo_previews'):p.mkdir(parents=True,exist_ok=True)
    data=mujoco.MjData(model)
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth=130
    camera.elevation=-19
    camera.distance=.80
    camera.lookat[:]=[frames[0,0],frames[0,1],.13]
    options=mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),18) if fontpath.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),13) if fontpath.exists() else ImageFont.load_default()
    mp4=videos/f'{name}.mp4';gif=gifs/f'{name}.gif'
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        with imageio.get_writer(mp4,fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
            for i,qpos in enumerate(frames):
                data.qpos[:]=qpos
                mujoco.mj_forward(model,data)
                # Camera tracks translation only; rendered poses are the recorded physics states.
                camera.lookat[:2]=qpos[:2]
                renderer.update_scene(data,camera=camera,scene_option=options)
                pixels=renderer.render()
                pic=Image.fromarray(pixels)
                draw=ImageDraw.Draw(pic)
                draw.rectangle((0,0,width,55),fill=(22,29,38))
                draw.text((16,8),TITLES[name],font=font,fill=(245,247,250))
                draw.text((16,33),'MuJoCo simulation | trained policy | 1x speed',font=small,fill=(179,194,210))
                draw.rectangle((0,height-26,width,height),fill=(22,29,38))
                draw.text((16,height-21),f'{i/25:04.2f}s  |  Microduck HD-1910',font=small,fill=(210,220,231))
                image=np.asarray(pic)
                writer.append_data(image)
                if i in (0,len(frames)//2,len(frames)-1):
                    pic.save(ROOT/'outputs/demo_previews'/f'{name}_{i:04}.png')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(mp4),
                    '-filter_complex','fps=20,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
                    '-loop','0',str(gif)],check=True)
    report['gif_fps']=20;report['gif_width']=480
    report['gif']=str(gif.relative_to(ROOT));report['mp4']=str(mp4.relative_to(ROOT))
    report['gif_bytes']=gif.stat().st_size;report['mp4_bytes']=mp4.stat().st_size
    report['gif_sha256']=hashlib.sha256(gif.read_bytes()).hexdigest()
    (media/f'{name}_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('skill','duration_s','final_state','reset_count','gif_bytes','mp4_bytes')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=list(TITLES))
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--skip-existing',action='store_true',help='Keep clips that already have a report')
    args=parser.parse_args()
    torch.set_float32_matmul_precision('high')
    for name in ([args.only] if args.only else TITLES):
        if args.skip_existing and all(f.exists() for f in (ROOT/'media'/f'{name}_report.json', ROOT/'media/gifs'/f'{name}.gif', ROOT/'media/videos'/f'{name}.mp4')):continue
        model,frames,report=record(name,args.seed)
        render(name,model,frames,report)


if __name__=='__main__':main()
