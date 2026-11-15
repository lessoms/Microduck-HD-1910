import torch
from torch import nn

def qmul(a,b):
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True)
    v=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)
    return torch.cat([w,v],-1)
def qconj(q):return torch.cat([q[...,:1],-q[...,1:]],-1)
def rotate(q,v):
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)
def normalize(q):return q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
def quat6(q):
    e1=torch.zeros_like(q[...,1:]);e1[...,0]=1
    e2=torch.zeros_like(e1);e2[...,1]=1
    return torch.cat([rotate(q,e1),rotate(q,e2)],-1)

class DifferentiableFK(nn.Module):
    """Compiled model tree, including colocated serial motor/backlash hinges."""
    def __init__(self,model):
        super().__init__();self.parents=model.body_parentid.tolist();self.root=next(i for i in range(model.nbody) if model.body(i).name.endswith('trunk_base'))
        descendants=[]
        for i in range(model.nbody):
            parent=i
            while parent and parent!=self.root:parent=self.parents[parent]
            if parent==self.root:descendants.append(i)
        self.robot_body_ids=descendants
        self.register_buffer('positions',torch.tensor(model.body_pos.copy(),dtype=torch.float32))
        self.register_buffer('quats',torch.tensor(model.body_quat.copy(),dtype=torch.float32))
        self.register_buffer('axes',torch.tensor(model.jnt_axis.copy(),dtype=torch.float32))
        self.register_buffer('anchors',torch.tensor(model.jnt_pos.copy(),dtype=torch.float32))
        self.register_buffer('q0',torch.tensor(model.qpos0.copy(),dtype=torch.float32))
        self.joints=[]
        for i in range(model.nbody):
            self.joints.append([(j,int(model.jnt_qposadr[j])-7) for j in range(int(model.body_jntadr[i]),int(model.body_jntadr[i]+model.body_jntnum[i])) if int(model.jnt_type[j])==3])
    def forward(self,root,joints):
        n=len(root);zero=torch.zeros((n,3),device=root.device,dtype=root.dtype)
        identity=torch.zeros((n,4),device=root.device,dtype=root.dtype);identity[:,0]=1
        ps=[zero];qs=[identity]
        for i in range(1,len(self.parents)):
            if i==self.root:p=root[:,:3];q=normalize(root[:,3:7])
            else:
                parent=self.parents[i];p=ps[parent]+rotate(qs[parent],self.positions[i].expand(n,-1));q=qmul(qs[parent],self.quats[i].expand(n,-1))
                for j,k in self.joints[i]:
                    angle=joints[:,k]-self.q0[k+7];half=angle[:,None]/2
                    r=torch.cat([torch.cos(half),torch.sin(half)*self.axes[j]],-1)
                    anchor=self.anchors[j].expand(n,-1)
                    p=p+rotate(q,anchor-rotate(r,anchor));q=qmul(q,r)
            ps.append(p);qs.append(q)
        return torch.stack(ps,1)[:,self.robot_body_ids],torch.stack(qs,1)[:,self.robot_body_ids]


def aligned_body_errors(root,refroot,body,body_quat,reference,align=False):
    """Yaw-only reference rotation; robot xy, reference z; retain global root goals."""
    pos=reference[...,:3];quat=normalize(reference[...,3:7])
    if align:
        def yaw(q):
            q=normalize(q);w,x,y,z=q.unbind(-1)
            angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))/2
            zero=torch.zeros_like(angle)
            return torch.stack([angle.cos(),zero,zero,angle.sin()],-1)
        delta=qmul(yaw(root[:,3:7]),qconj(yaw(refroot[:,3:7])))
        anchor=torch.cat([root[:,:2],refroot[:,2:3]],-1)
        rotation=delta[:,None].expand_as(quat)
        pos=anchor[:,None]+rotate(rotation,pos-refroot[:,None,:3])
        quat=qmul(rotation,quat)
    rel=qmul(qconj(normalize(body_quat)),quat)
    angle=2*torch.atan2(rel[...,1:].norm(dim=-1),rel[...,0].abs().clamp_min(1e-8))
    return (body-pos).square().sum(-1).mean(-1),angle.square().mean(-1)
