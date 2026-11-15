"""Bit-identical SuperMimic inference actor, extracted without training modules."""
import torch
from torch import nn

def elu_mlp(inputs,hidden,outputs,small_output=False):
    layers=[]
    for width in hidden:layers.extend([nn.Linear(inputs,width),nn.ELU()]);inputs=width
    layers.append(nn.Linear(inputs,outputs))
    if small_output:
        nn.init.orthogonal_(layers[-1].weight,gain=.01);nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)

class OldAlignedPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.target_encoder=elu_mlp(440,[2048,1024,512,256],32)
        self.decoder=elu_mlp(480+32,[2048,1024,512,256],14,small_output=True)
        self.register_buffer('observation_scale',torch.tensor([.2]*3+[1.]*3+[1.]*14+[.1]*14+[1.]*14).repeat(10))
    def forward(self,history,future):
        assert history.shape[-1]==480 and future.shape[-1]==440
        z=torch.tanh(self.target_encoder(future.clamp(-10,10)))
        # Current action bounds remain unchanged; old task/physics are not substituted.
        return torch.tanh(self.decoder(torch.cat([(history*self.observation_scale).clamp(-10,10),z],-1)))
