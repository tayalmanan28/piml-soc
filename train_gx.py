import torch
from dynamics.Boat2DAug import Boat2DAug
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.modules import BCNetwork

if __name__ =="__main__":

    dyn = Boat2DAug()

    num_samples = 100

    xs = torch.linspace(-3, 2, num_samples)
    ys = torch.linspace(-2, 2, num_samples)
    zs = torch.linspace(-0.1, 14.86, num_samples)
    coords = torch.cartesian_prod(xs, ys, zs)

    lx = dyn.boundary_fn(coords)


    model = BCNetwork()
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    coords=coords.cuda().float()
    lx = lx.cuda().float()
    num_epochs = 10000
    state_mean=torch.tensor([-0.5, 0.0, 7.38])
    state_var=torch.tensor([2.5, 2.0, 7.48])

    for epoch in range(num_epochs):#tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        model.train()
        train_loss = 0.0
        for i in range(10):
            optimizer.zero_grad()
            # idx = torch.randperm(coords.size(0))[:2000]
            # outputs = model(coords[idx,...].unsqueeze(0).clone())
            # labels = lx[idx].clone()
            # sample coords
            coords_samples =(torch.zeros(1, 2000, 3).uniform_(-1, 1) * state_var
                            ) + state_mean
            # sample switch states
            labels = dyn.boundary_fn(coords_samples).cuda().float()
            outputs = model(coords_samples.clone().cuda())

            loss = criterion(outputs.flatten(), labels)

            # print(outputs.shape,labels.shape)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            # print(i,loss.item(),train_loss)

        train_loss /= 10
        if epoch%100==0:
            print('Epoch:', epoch,'Train Loss:',train_loss)
            
    lx2=model(coords).detach().cpu()
    print('static data loss:',torch.sum(torch.abs(lx-lx2.cuda()))/1000000)
    torch.save(model.state_dict(), 'Boat_gx.pth')


