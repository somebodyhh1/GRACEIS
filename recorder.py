import torch
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

class Recorder():
    def __init__(self) -> None:

        self.all_directions=[]
        self.Lambda_1s=[]
        self.positive_theta_rate=[]
    def record_direction(self,direction,Lambda_1,theta):
        self.all_directions.append(direction)
        print("Lambda1 shape",Lambda_1.shape)
        self.Lambda_1s.append(Lambda_1.diag())
        idx=torch.where(theta>0)
        pos_size=idx[0].shape[0]
        size=theta.shape[0]
        pos_rate=pos_size/size
        self.positive_theta_rate.append(pos_rate)
        
    def show_directions(self,Lambda):
        plt.figure(figsize=(10,3))
        num_of_slice=5
        num_of_eigs=Lambda.shape[0]
        size_of_slice=(int)(num_of_eigs/num_of_slice)
        inf=100000000
        indices=[]
        Lambda1=Lambda.clone()
        directions=torch.stack(self.all_directions,dim=1).T
        Lambda_1s=torch.stack(self.Lambda_1s,dim=1).T
        print(directions.shape)
        x_axis=np.array(list(range(len(self.all_directions))))
        for i in range(num_of_slice):
            _,indice=Lambda1.topk(size_of_slice)
            indices.append(indice)
            slice_direction=directions[:,indice]
            slice_direction=slice_direction.sum(dim=1)
            slice_lambda_1=Lambda_1s[:,indice].mean(dim=1)
            plt.subplot(131)
            plt.plot(x_axis,slice_direction.cpu(),label="top "+str(20*i)+"-"+str(20*(i+1))+r"% $\lambda$")
            change_amplitude=torch.mean(Lambda1[indice])*slice_direction
            plt.subplot(132)
            plt.plot(x_axis,change_amplitude.cpu(),label="top "+str(20*i)+"-"+str(20*(i+1))+r"% $\lambda$")
            plt.subplot(133)
            plt.plot(x_axis,slice_lambda_1.cpu(),label="top "+str(20*i)+"-"+str(20*(i+1))+r"% $\lambda$",linewidth=2)
            Lambda1[indice]=-inf
        
        plt.subplot(131)
        plt.legend(fontsize=7)
        plt.xlabel("(a))")
        plt.grid()
        plt.subplot(132)
        plt.legend(fontsize=7)
        plt.xlabel("(b))")
        plt.grid()
        plt.subplot(133)
        plt.legend(fontsize=7)
        plt.xlabel("(c))")
        plt.grid()
        plt.tight_layout()
        plt.savefig('images/theta.svg')
        plt.close()