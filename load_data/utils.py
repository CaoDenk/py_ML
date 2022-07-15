from turtle import color
import torch
from matplotlib import pyplot as plt

def plot_image(image, lable, title):
    
    fig=plt.figure()
    for i in range(6):
        fig.add_subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(image[i][0]*0.3081+0.1307, cmap='gray',interpolation='none')
        plt.title("{}:{}".format(title,lable[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_curve(data):
    fig=plt.figure()
    fig.plot(range(len(data)),data,color='blue')
    plt.legend('step')
    plt.xlabel('step')
    plt.ylabel('value') 
    plt.show()

def one_hot(lable,depth=10):
    out=torch.zeros(lable.size(0),depth)
    idex=torch.LongTensor(lable).view(-1,1)
    out.scatter_(dim=1,index=idex,value=1)
    return out