"""Episode reset quotas, preserving the legacy uniform RNG path by default."""
import torch

class ResetFrameSampler:
    def __init__(self,spec=None,device='cpu',frame_count=650):
        self.frame_count=frame_count;self.bin_count=(frame_count+49)//50
        self.device=device;self.spec=spec;self.frame_counts=torch.zeros(frame_count,dtype=torch.long,device=device)
        if spec is None:self.groups=[dict(name='uniform',probability=1.,first=0,last_exclusive=frame_count)]
        elif spec['mode'] in ('adaptive_return','adaptive_window'):
            quotas = [('uniform', .5), ('hard', .4), ('memory', .1)]
            if spec['mode'] == 'adaptive_window' and spec.get('split_current_difficulty'):
                quotas = [('uniform', .5), ('tracking_drop', .25), ('hard', .15), ('memory', .1)]
                self.frame_drop_score = torch.zeros(frame_count, device=device)
            self.groups=[dict(name=n,probability=p,first=0,last_exclusive=frame_count) for n,p in quotas]
            self.bin_return=torch.full((self.bin_count,),float(spec.get('g_max',1.)),device=device)
            self.bin_memory=torch.full((self.bin_count,),1/self.bin_count,device=device)
        else:
            assert spec['mode']=='mixture';self.groups=spec['groups']
            assert abs(sum(float(g['probability']) for g in self.groups)-1.)<1e-8
            assert all(0<=g['first']<g['last_exclusive']<=frame_count and g['probability']>0 for g in self.groups)
        self.probabilities=torch.tensor([g['probability'] for g in self.groups],dtype=torch.float64,device=device)
        self.group_counts=torch.zeros(len(self.groups),dtype=torch.long,device=device)
    def draw(self,count,generator):
        if self.spec is None:
            frames=torch.randint(0,self.frame_count,(count,),device=self.device,generator=generator);selected=torch.zeros(count,dtype=torch.long,device=self.device)
        elif self.spec['mode'] in ('adaptive_return','adaptive_window'):
            hard=self.hard_distribution()
            self.bin_memory.lerp_(hard,self.spec.get('memory_alpha',.01))
            selected=torch.multinomial(self.probabilities,count,replacement=True,generator=generator)
            distributions=torch.stack([torch.full_like(hard,1/self.bin_count),hard,self.bin_memory/self.bin_memory.sum()])
            split = hasattr(self, 'frame_drop_score')
            # The tracking-drop group falls back to ordinary difficulty until a drop is observed.
            distribution_index = selected - (selected >= 2).long() if split else selected
            bins=torch.multinomial(distributions[distribution_index],1,generator=generator).squeeze(-1)
            if self.frame_count % 50:
                widths=(self.frame_count-bins*50).clamp(max=50)
                offset=(torch.rand(count,device=self.device,generator=generator)*widths).long()
            else:
                offset=torch.randint(0,50,(count,),device=self.device,generator=generator)
            frames=bins*50+offset
            if self.frame_count % 50 or split:
                uniform = torch.randint(
                    0, self.frame_count, (count,), device=self.device, generator=generator
                )
                frames = torch.where(selected == 0, uniform, frames)
            if split:
                transition = selected == 1
                if transition.any() and self.frame_drop_score.any():
                    frames[transition] = torch.multinomial(
                        self.frame_drop_score, int(transition.sum()), replacement=True,
                        generator=generator,
                    )
        else:
            selected=torch.multinomial(self.probabilities,count,replacement=True,generator=generator);frames=torch.empty(count,dtype=torch.long,device=self.device)
            for i,g in enumerate(self.groups):
                mask=selected==i;n=int(mask.sum())
                if n:
                    frames[mask]=g['first'] if g['last_exclusive']==g['first']+1 else torch.randint(g['first'],g['last_exclusive'],(n,),device=self.device,generator=generator)
        self.group_counts+=torch.bincount(selected,minlength=len(self.groups));self.frame_counts+=torch.bincount(frames,minlength=self.frame_count)
        return frames
    def hard_distribution(self):
        if self.spec['mode']=='adaptive_return':return torch.softmax(-25*self.bin_return/self.spec['g_max'],0)
        raw=torch.softmax(-self.bin_return/self.spec['temperature'],0)
        # Water filling: fix saturated bins, redistribute only among free bins.
        p=torch.zeros_like(raw);free=torch.ones_like(raw,dtype=torch.bool);remaining=1.
        for _ in range(len(raw)):
            candidate=raw[free]/raw[free].sum()*remaining
            over=candidate>self.spec['hard_probability_cap']
            if not over.any():p[free]=candidate;break
            saturated=free.nonzero().flatten()[over]
            p[saturated]=self.spec['hard_probability_cap'];free[saturated]=False
            remaining=1-p.sum()
        return p
    def update_episode(self,frames,rewards,terminated=False):
        if self.spec is not None and self.spec['mode']=='adaptive_window':
            scores=rewards.clamp(0,1);n=len(scores)
            start=torch.arange(n,device=scores.device)
            target=torch.minimum(torch.full_like(start,50),self.frame_count-frames)
            end=torch.minimum(start+target,torch.full_like(start,n))
            sums=torch.cat([scores.new_zeros(1),scores.cumsum(0)])
            values=(sums[end]-sums[start])/target
            valid=(end-start==target)|bool(terminated)
            bins=frames.clamp(max=self.frame_count-1)//50
            for b in bins[valid].unique():
                self.bin_return[b].lerp_(values[valid&(bins==b)].mean(),self.spec.get('ema_alpha',.1))
            if hasattr(self, 'frame_drop_score'):
                # Require a full past second and room for a full future second
                # in the reference. Only physical failures zero-pad the future.
                eligible = (start >= 50) & (target == 50) & valid
                boundaries = start[eligible]
                previous = (sums[boundaries]-sums[boundaries-50])/50
                drops = previous * (previous-values[eligible]).clamp_min(0)
                indices = frames[eligible]
                self.frame_drop_score[indices] = self.frame_drop_score[indices].lerp(
                    drops, self.spec.get('ema_alpha', .1)
                )
            return

        if self.spec is None or self.spec['mode']!='adaptive_return':return
        returns=torch.empty_like(rewards);g=0.
        for i in reversed(range(len(rewards))):
            g=rewards[i]+self.spec.get('gamma',.99)*g;returns[i]=g
        bins=frames.clamp(max=self.frame_count-1)//50
        for b in bins.unique():
            self.bin_return[b].lerp_(returns[bins==b].mean(),self.spec.get('ema_alpha',.1))
    def state_dict(self):
        state=dict(spec=self.spec,group_counts=self.group_counts.clone(),frame_counts=self.frame_counts.clone())
        if hasattr(self,'bin_return'):state.update(bin_return=self.bin_return.clone(),bin_memory=self.bin_memory.clone())
        if hasattr(self, 'frame_drop_score'):
            state['frame_drop_score'] = self.frame_drop_score.clone()
        return state
    def load_state_dict(self,state):
        assert state['spec']==self.spec
        self.group_counts.copy_(state['group_counts']);self.frame_counts.copy_(state['frame_counts'])
        if hasattr(self,'bin_return'):
            self.bin_return.copy_(state['bin_return']);self.bin_memory.copy_(state['bin_memory'])
        if hasattr(self, 'frame_drop_score'):
            self.frame_drop_score.copy_(state['frame_drop_score'])
    def report(self):
        count=int(self.group_counts.sum());result=dict(total_sampled_resets=count,groups=[dict(name=g['name'],configured_probability=g['probability'],count=int(self.group_counts[i]),observed_fraction=float(self.group_counts[i])/max(1,count),first=g['first'],last_exclusive=g['last_exclusive']) for i,g in enumerate(self.groups)],exact_frame0_count=int(self.frame_counts[0]),explicit_frame_evaluation_resets_excluded=True)
        if hasattr(self,'bin_return'):result.update(bin_return=self.bin_return.tolist(),bin_memory=self.bin_memory.tolist(),bin_seconds=1,hard_distribution=self.hard_distribution().tolist(),score_semantics='fixed50-step normalized tracking mean' if self.spec['mode']=='adaptive_window' else 'discounted return-to-go')
        return result

def validate_sampling_fork(old_cfg,new_cfg,old_collection,new_collection):
    allowed={'rounds','evaluation_every','collection_config'}
    assert {k:v for k,v in old_cfg.items() if k not in allowed}=={k:v for k,v in new_cfg.items() if k not in allowed},'Unexpected algorithm/network change in reset experiment'
    allowed_collection={'max_episode_steps','reset_sampling','initialization'}
    if new_collection.get('body_reference_alignment')=='robot_xy_yaw':
        allowed_collection|={'body_reference_alignment','reward'}
        old_reward=old_collection['reward'].copy();new_reward=new_collection['reward'].copy()
        if 'body_orientation' not in old_reward:new_reward.pop('body_orientation',None)
        assert old_reward==new_reward,'Only adding body orientation reward is allowed'
    assert {k:v for k,v in old_collection.items() if k not in allowed_collection}=={k:v for k,v in new_collection.items() if k not in allowed_collection},'Unexpected physics/reward/reference change'
