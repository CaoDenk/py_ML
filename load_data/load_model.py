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

model = Net()
model.load_state_dict(torch.load("mnist_net.pth"))
model.eval()

x,y=next(iter(test_loader))
out=model(x.view(-1,28*28))
pred=out.argmax(dim=1)
plot_image(x,pred,'pred') 