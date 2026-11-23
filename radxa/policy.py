"""CPU actor inference; no MuJoCo, mjlab, CUDA or original project imports."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, weights, prefix=''):
        super().__init__()
        for name in ('target_encoder','decoder'):
            # Landing prefix contains a dot; use suffix for layer index.
            pairs=sorted((int(k[len(prefix+name)+1:].split('.')[0]),v) for k,v in weights.items() if k.startswith(prefix+name+'.') and k.endswith('.weight'))
            layers=[]
            for i,(_,w) in enumerate(pairs):
                layers.append(nn.Linear(w.shape[1],w.shape[0]))
                if i+1<len(pairs):layers.append(nn.ELU())
            setattr(self,name,nn.Sequential(*layers))
        self.register_buffer('observation_scale',weights[prefix+'observation_scale'].clone())
        own={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix) and not k[len(prefix):].startswith('landing_policy.')}
        self.load_state_dict(own)
        self.eval().requires_grad_(False)
    def forward(self,history,future):
        z=torch.tanh(self.target_encoder(future.clamp(-10,10)))
        return torch.tanh(self.decoder(torch.cat(((history*self.observation_scale).clamp(-10,10),z),-1)))

class Policies:
    def __init__(self,model,assets):
        torch.set_num_threads(1)
        self.bundle=torch.load(model,weights_only=True,map_location='cpu',mmap=True)
        self.refs=np.load(assets/'references.npz',allow_pickle=False)
        import json
        self.meta=json.loads((assets/'metadata.json').read_text())
        self.actors={};self.landing=None
        for skill,saved in self.bundle['skills'].items():
            w=saved['models']['actor' if skill=='grasp' else 'policy']
            self.actors[skill]=Actor(w)
            if skill=='flamingo' and any(k.startswith('landing_policy.') for k in w):self.landing=Actor(w,'landing_policy.')
    def future(self,skill,frame,command):
        if skill=='velocity':
            x=np.zeros((10,44),np.float32);x[:,:3]=command;return x.reshape(1,440)
        return self.refs[skill+'_future'][min(frame,len(self.refs[skill+'_future'])-1)][None]
    def bound(self,skill,raw):
        m=self.meta['skills'][skill];home=np.array(m['home']);limits=np.array(m['limits'])
        offset=np.clip(raw*m['scale'],-m['clip'],m['clip'])
        return np.clip(home+offset,limits[:,0]+m['margin'],limits[:,1]-m['margin'])-home
    @torch.inference_mode()
    def action(self,skill,frame,history,command,previous):
        h=torch.as_tensor(history.reshape(1,480),dtype=torch.float32)
        f=torch.as_tensor(self.future(skill,frame,command).copy(),dtype=torch.float32)
        out=self.bound(skill,self.actors[skill](h,f).numpy()[0])
        if skill=='flamingo' and self.landing is not None:
            cfg=self.meta['skills'][skill]['landing'];start=cfg['start_frame']
            if frame>=start:
                low=self.bound(skill,self.landing(h,f).numpy()[0]);phase=np.clip((frame-start)/cfg['blend_ticks'],0,1);weight=phase*phase*(3-2*phase)
                desired=out+(low-out)*weight
                out=previous+np.clip(desired-previous,-cfg['max_delta_rad'],cfg['max_delta_rad'])
        return out.astype(np.float32)
