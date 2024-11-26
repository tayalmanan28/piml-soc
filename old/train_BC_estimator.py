import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
mat = scipy.io.loadmat('gait_from_clf_qp_ral.mat')
gait=mat['xRec'][:, [2, 3, 6, 7]]

scale=torch.tensor([1,1,0.1,0.1])
gait_scaled=torch.tensor(gait)*scale
distance_threshold=0.05

lL = 1
mL = 1
g = 9.81
mH = 1
umax=4.0
def get_fvec(state_condensed):
    q1 = state_condensed[...,0]
    q2 = state_condensed[...,1]
    dq1 = state_condensed[...,2]
    dq2 = state_condensed[...,3]


    t2 = torch.cos(q1)
    t3 = torch.sin(q1)
    t4 = torch.sin(q2)
    t5 = dq1**2
    t6 = dq2**2
    t7 = mH*4.0
    t8 = mL*3.0
    t9 = q2*2.0
    t12 = 1.0/lL
    t10 = torch.cos(t9)
    t11 = torch.sin(t9)
    t13 = g*t3*t7
    t14 = g*mL*t3*4.0
    t17 = dq1*dq2*lL*mL*t4*4.0
    t18 = lL*mL*t4*t6*2.0
    t15 = mL*t10*2.0
    t16 = -t15
    t19 = t7+t8+t16
    t20 = 1.0/t19
    fvec=torch.zeros_like(state_condensed)
    fvec[...,0]=dq1
    fvec[...,1]=dq2
    fvec[...,2]=-t12*t20*(-t14+t17+t18-g*mH*t3*4.0+g*mL*torch.sin(q1+t9)*2.0+lL*mL*t4*t5*2.0-lL*mL*t5*t11*2.0)
    fvec[...,3]=-t12*t20*(t13+t14-t17-t18+g*mH*t2*t4*8.0+g*mL*t2*t4*1.0e+1-g*mL*t2*t11*2.0-g*mL*t3*t10*2.0-lL*mH*t4*t5*8.0-lL*mL*t4*t5*1.2e+1+lL*mL*t5*t11*4.0+lL*mL*t6*t11*2.0-g*mL*t3*torch.cos(q2)*2.0+dq1*dq2*lL*mL*t11*4.0)

    return fvec

def get_gvec(state_condensed):

    q2 = state_condensed[...,1]

    t2 = torch.cos(q2)
    t3 = mH*4.0
    t4 = mL*5.0
    t6 = 1.0/lL**2
    t5 = t2**2
    t7 = mL*t5*4.0
    t8 = -t7
    t9 = t3+t4+t8
    t10 = 1.0/t9
    gvec = torch.zeros(1, state_condensed.shape[1],4)
    gvec[...,2] = t6*t10*(t2*8.0-4.0)
    gvec[...,3] = (t6*t10*(mH*1.6e+1+mL*2.4e+1-mL*t2*1.6e+1))/mL
    return gvec

def dsdt(x,u):
    dx = get_fvec(x.clone()) + get_gvec(x.clone()) * torch.cat([torch.zeros_like(u),u],dim=-1) 
    return dx

def boundary_fn(xs):
    xs_scaled=(xs*scale).squeeze(0).repeat(gait_scaled.shape[0],1,1)
    distances_to_gait=torch.norm(xs_scaled-gait_scaled.unsqueeze(1).repeat(1,xs_scaled.shape[1],1),dim=-1)
    distance_to_gait,_ = torch.min(distances_to_gait,dim=0)
    l_x = distance_to_gait - distance_threshold
    return l_x.squeeze(0)

# reset map is applied when q1 <=0 and q2 = -2*q1 and 2dq1 + dq2 <= 0
# [−0.52, 0.52] × [−1.04, 1.04] × [−4, 4] × [−8, 8],
def sample_switch_state(num_points=100):
    xs_pre_switch=torch.rand(1,num_points,4)
    xs_pre_switch[...,0]=xs_pre_switch[...,0]*-0.52
    xs_pre_switch[...,1]=-2.0*xs_pre_switch[...,0]
    xs_pre_switch[...,2]=xs_pre_switch[...,2]*8.0-4.0
    xs_pre_switch[...,3]= -8.0 + (-2*xs_pre_switch[...,2]+8.0)*xs_pre_switch[...,3]
    return xs_pre_switch

def reset_map_condensed(xs_pre):
    xs_post=torch.zeros_like(xs_pre)
    xs_post[...,0]=xs_pre[...,0]+xs_pre[...,1]
    xs_post[...,1]=-xs_pre[...,1]
    dq_post_impact=dq_post_impact_condensed(xs_pre.clone())
    xs_post[...,2] = dq_post_impact[...,0]+dq_post_impact[...,1]
    xs_post[...,3]= -dq_post_impact[...,1]
    return xs_post 

def dq_post_impact_condensed(xs):
    q1 = xs[...,0]
    q2 = xs[...,1]
    dq1 = xs[...,2]
    dq2 = xs[...,3]

    dx = -torch.cos(q1) * dq1
    dy = -torch.sin(q1) * dq1

    t2 = torch.cos(q1)
    t3 = torch.cos(q2)
    t4 = torch.sin(q1)
    t5 = q1+q2
    t7 = q2*2.0
    t14 = -q2
    t9 = torch.cos(t7)
    t12 = torch.cos(t5)
    t13 = torch.sin(t5)
    t15 = dq1*t3*2.0
    t16 = dq2*t3*2.0
    t17 = q2+t5
    t21 = dx*t2*8.0
    t22 = q1+t14
    t23 = dy*t4*8.0
    t18 = torch.cos(t17)
    t19 = torch.sin(t17)
    t20 = t9*2.0
    t25 = -t21
    t26 = -t23
    t27 = dq1*t20
    t30 = t20-7.0
    t31 = dx*t18*1.0e+1
    t32 = dy*t19*1.0e+1
    t33 = 1.0/t30
    dq_post_impact = torch.cat(((t33*(dq1*-7.0+t15+t16+t25+t26+t27+t31+t32))[...,None],
        (-t33*(dq1*-8.0-dq2+t15+t16+t25+t26+t27+t31+t32-dx*t12*8.0-dy*t13*8.0+dx*torch.cos(t22)*2.0+dy*torch.sin(t22)*2.0))[...,None]),dim=-1)
    return dq_post_impact

xs_pre_switch=sample_switch_state()
control=torch.rand_like(xs_pre_switch)[...,:2]
xs_post=reset_map_condensed(xs_pre_switch)
# print(xs_pre_switch)
# print(xs_post)
# print(boundary_fn(xs_pre_switch))
# print(boundary_fn(xs_post))
# print(dsdt(xs_pre_switch,control))
plt.figure(figsize=(4,7))
plt.scatter(xs_pre_switch[...,0],xs_pre_switch[...,1],color='b')
plt.scatter(xs_post[...,0],xs_post[...,1],color='r')
plt.plot(gait_scaled[...,0],gait_scaled[...,1])

xs = torch.linspace(-0.52, 0.52, 30)
ys = torch.linspace(-1.04, 1.04, 30)
dq1s = torch.linspace(-4.0, 4.0, 30)
dq2s = torch.linspace(-8.0, 8.0, 30)
coords = torch.cartesian_prod(xs, ys, dq1s, dq2s)

lx=boundary_fn(coords)
brt_counts=torch.sum(torch.sum(lx.reshape(30,30,30,30)<=0,axis=-1),axis=-1).T.float()


BRT_img = brt_counts.numpy()
imshow_kwargs = {
    'vmax': 10,
    'vmin': 0,
    'cmap': 'RdYlBu',
    'extent': (-0.52, 0.52, -1.04, 1.04),
    'origin': 'lower',
}
fig= plt.figure()
ax = fig.add_subplot(1, 1, 1)
s1 = ax.imshow(BRT_img, **imshow_kwargs)
fig.colorbar(s1)

plt.plot([0,-0.52],[0,1.04],'r')
plt.plot(gait_scaled[...,0],gait_scaled[...,1],'b')
# print(coords.shape)




plt.show()
class BCNetwork(nn.Module):
    def __init__(self, input_dim=4, num_nl=64):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_nl)

        self.fc2 = nn.Linear(num_nl, num_nl)

        self.fc3 = nn.Linear(num_nl, num_nl)

        self.fc4 = nn.Linear(num_nl, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)

model = BCNetwork()
model.load_state_dict(
                torch.load('BC_estimator.pth', map_location='cpu'))
model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

coords=coords.cuda().float()
lx=lx.cuda().float()
num_epochs = 0
state_mean=torch.tensor([0.0, 0.0, 0.0, 0.0])
state_var=torch.tensor([0.52, 1.04, 4.0, 8.0])

for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
    model.train()
    train_loss = 0.0
    for i in range(10):
        optimizer.zero_grad()
        # idx = torch.randperm(coords.size(0))[:2000]
        # outputs = model(coords[idx,...].unsqueeze(0).clone())
        # labels = lx[idx].clone()
        # sample coords
        coords_samples =(torch.zeros(1, 2000, 4).uniform_(-1, 1) * state_var
                          ) + state_mean
        # sample switch states
        xs_pre_switch=sample_switch_state(500)
        coords_samples=torch.cat([coords_samples,xs_pre_switch],dim=1).float()
        labels=boundary_fn(coords_samples).cuda().float()
        outputs = model(coords_samples.clone().cuda())

        # compute l(x)
        outputs[outputs<0]=outputs[outputs<0]*50
        labels[labels<0]=labels[labels<0]*50
        # print(outputs.shape,labels.shape)
        loss = criterion(outputs.flatten(), labels)

        # print(outputs.shape,labels.shape)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * 2500
        # print(i,loss.item(),train_loss)

    train_loss /= 10
    if epoch%100==0:
        print(train_loss)
        torch.save(model.state_dict(), 'BC_estimator_%04d.pth'% epoch)
        lx2=model(coords).detach().cpu()
        print(torch.sum(torch.abs(lx-lx2.cuda()))/810000)


coords=coords
lx2=model(coords).detach().cpu()
brt_counts=torch.sum(torch.sum(lx2.reshape(30,30,30,30)<=0,axis=-1),axis=-1).T.float()
BRT_img = brt_counts.numpy()

fig= plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
s1 = ax.imshow(BRT_img, **imshow_kwargs)
fig.colorbar(s1)

plt.plot([0,-0.52],[0,1.04],'r')
plt.plot(gait_scaled[...,0],gait_scaled[...,1],'b')
print(torch.sum(torch.abs(lx-lx2.cuda()))/810000)
plt.show()
torch.save(model.state_dict(), 'BC_estimator.pth')