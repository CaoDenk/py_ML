
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, one_hot
batch_size=64
train_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',
            train=True,download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',
            train=False,download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=batch_size,shuffle=True)

x,label =next(iter(train_loader))
print(x.shape,label.shape)
plot_image(x,label,'sample')
# plt.plot(sample[0][0,:])


class Net(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,10)
    
    def forward(self,x):
        
 
        # x=x.view(-1,28*28)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        
        return x


net =Net()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
for epoch in range(3):

    for batch_idex,(x,y) in enumerate(train_loader):
        # x=x.view(-1,28*28)
        # y=one_hot(y,10)
        # print(x.shape,y.shape)
        # break
        # print(x.shape,y.shape)
        x=x.view(-1,28*28)
        out=net(x)
        
        y_onehot=one_hot(y)
        loss=F.mse_loss(out,y_onehot)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step() # update weights

        if batch_idex%10==0:
            print(epoch,batch_idex,loss.item())

total_correct=0
for x,y in test_loader:
    x=x.view(-1,28*28)
    out=net(x)
    pred=out.argmax(dim=1)
    correct=pred.eq(y).sum().float().item()
    total_correct+=correct

total_num=len(test_loader.dataset) 
acc=total_correct/total_num
print("test_acc",acc)


# net=torch.load("mnist_net.pth")

x,y=next(iter(test_loader))
out=net(x.view(-1,28*28))
pred=out.argmax(dim=1)
plot_image(x,pred,'pred') 


# state={}
# state['model_state'] = net.state_dict()
# state['loss'] = loss
# state['e'] = epoch
# state['optimizer'] = optimizer.state_dict()
# torch.save(state, "mnist_net.pth")
torch.save(net.state_dict(),'mnist_net.pth')
# print()