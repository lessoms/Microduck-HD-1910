"""Ball-aware difficulty; original random group quotas; four one-second bins."""
import torch
from microduck_sm.reset_sampling import ResetFrameSampler


def combined_difficulty_score(robot, ball, failed, robot_weight=.7, ball_weight=.3):
    task=torch.where(failed,torch.zeros_like(ball),ball.clamp(0,1))
    return (robot_weight*robot.clamp(0,1)+ball_weight*task).clamp(0,1)


def kick_attempt_eligible(frame, ball_speed, kick_end=50, max_initial_speed=.05):
    return (frame<kick_end)&(ball_speed<max_initial_speed)


def qualified_kick_motion(right_touch,left_touch,movement,minimum=.03):
    return right_touch&~left_touch&(movement>=minimum)


class BallDifficultySampler(ResetFrameSampler):
    """Four capped bins and 50-step scoring windows; no new stage sampling quota."""
    def __init__(self,spec,device='cpu',frame_count=200):
        if spec['mode']!='adaptive_window' or not spec.get('split_current_difficulty'):
            raise ValueError('Ball difficulty requires adaptive_window with tracking-drop group')
        super().__init__(spec,device,frame_count)
        self.failure_credit=torch.zeros(frame_count,device=device)
        self.edges=torch.cat((torch.arange(0,frame_count,50,device=device),torch.tensor([frame_count],device=device))).long()
        if self.bin_count!=4:raise ValueError('4s ball difficulty requires four bins')
        self.eligible_attempts=0;self.continuation_resets=0;self.failed_kicks=0;self.successful_kicks=0
    def bins(self,frames):
        return torch.bucketize(frames.contiguous(),self.edges[1:-1],right=True)
    def draw(self,count,generator):
        hard=self.hard_distribution()
        self.bin_memory.lerp_(hard,self.spec.get('memory_alpha',.01))
        selected=torch.multinomial(self.probabilities,count,replacement=True,generator=generator)
        distributions=torch.stack((torch.full_like(hard,1/self.bin_count),hard,
                                   self.bin_memory/self.bin_memory.sum()))
        indices=selected-(selected>=2).long()
        bins=torch.multinomial(distributions[indices],1,generator=generator).squeeze(-1)
        widths=self.edges[bins+1]-self.edges[bins]
        frames=self.edges[bins]+(torch.rand(count,device=self.device,generator=generator)*widths).long()
        uniform=torch.randint(self.frame_count,(count,),device=self.device,generator=generator)
        frames=torch.where(selected==0,uniform,frames)
        # Existing tracking-drop group also carries explicit failed-kick credit.
        transition=selected==1
        if transition.any() and self.frame_drop_score.any():
            frames[transition]=torch.multinomial(self.frame_drop_score,int(transition.sum()),replacement=True,generator=generator)
        self.group_counts+=torch.bincount(selected,minlength=len(self.groups))
        self.frame_counts+=torch.bincount(frames,minlength=self.frame_count)
        return frames
    def update_episode(self,frames,scores,terminated=False):
        scores=scores.clamp(0,1);n=len(scores);start=torch.arange(n,device=self.device)
        # The difficulty window remains one second, independently of bin width.
        target=torch.minimum(torch.full_like(start,50),self.frame_count-frames)
        end=torch.minimum(start+target,torch.full_like(start,n))
        sums=torch.cat((scores.new_zeros(1),scores.cumsum(0)))
        values=(sums[end]-sums[start])/target
        valid=(end-start==target)|bool(terminated)
        bins=self.bins(frames.clamp(max=self.frame_count-1))
        for b in bins[valid].unique():
            self.bin_return[b].lerp_(values[valid&(bins==b)].mean(),self.spec.get('ema_alpha',.1))
        eligible=(start>=50)&(target==50)&valid
        boundaries=start[eligible]
        previous=(sums[boundaries]-sums[boundaries-50])/50
        drops=previous*(previous-values[eligible]).clamp_min(0)
        which=frames[eligible]
        # A later smooth-pose window must not erase the explicit missed-kick signal.
        updated=self.frame_drop_score[which].lerp(drops,self.spec.get('ema_alpha',.1))
        self.frame_drop_score[which]=torch.maximum(updated,self.failure_credit[which])
    def mark_kick_failure(self,frames):
        frames=frames.unique()
        if not len(frames):return
        self.failed_kicks+=1
        alpha=self.spec.get('ema_alpha',.1)
        # Advanced indexing returns a copy; assign the updated values explicitly.
        self.failure_credit[frames]=self.failure_credit[frames].lerp(torch.ones_like(self.failure_credit[frames]),alpha)
        self.frame_drop_score[frames]=torch.maximum(self.frame_drop_score[frames],self.failure_credit[frames])
        for b in self.bins(frames).unique():self.bin_return[b].lerp_(torch.zeros_like(self.bin_return[b]),alpha)
    def mark_kick_success(self,frames):
        self.successful_kicks+=1
        alpha=self.spec.get('ema_alpha',.1)
        self.failure_credit[frames]=self.failure_credit[frames]*(1-alpha)
        self.frame_drop_score[frames]=self.frame_drop_score[frames]*(1-alpha)
    def record_initializations(self,eligible):
        self.eligible_attempts+=int(eligible.sum());self.continuation_resets+=int((~eligible).sum())
    def state_dict(self):
        result=super().state_dict()
        result.update(bin_edges=self.edges.clone(),failure_credit=self.failure_credit.clone(),
            eligible_attempts=self.eligible_attempts,continuation_resets=self.continuation_resets,
            failed_kicks=self.failed_kicks,successful_kicks=self.successful_kicks)
        return result
    def load_state_dict(self,state):
        if not torch.equal(state['bin_edges'].to(self.device),self.edges):raise ValueError('Difficulty bins changed')
        super().load_state_dict(state)
        self.failure_credit.copy_(state['failure_credit'])
        for k in ['eligible_attempts','continuation_resets','failed_kicks','successful_kicks']:setattr(self,k,state[k])
    def report(self):
        result=super().report()
        result.update(bin_edges=self.edges.tolist(),bin_seconds=1,bin_durations_seconds=((self.edges[1:]-self.edges[:-1])/50).tolist(),
            window_control_steps=50,score_semantics='0.7 robot tracking + 0.3 ball tracking; missed-kick credit to preceding kick frames',
            eligible_attempts=self.eligible_attempts,continuation_resets=self.continuation_resets,
            failed_kicks=self.failed_kicks,successful_kicks=self.successful_kicks,
            failure_credit=self.failure_credit.tolist())
        return result
