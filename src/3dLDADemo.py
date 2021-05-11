import NPLDA
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

colours = ['r','g','b','purple']
names = ['Red','Green','Blue','Purple']
n = 100
data = [
    np.array([np.random.rand(n)*10-5,np.random.rand(n)*10+10,np.random.rand(n)*10]).T,
    np.array([np.random.rand(n)*10-15,np.random.rand(n)*10-10,np.random.rand(n)*10-10]).T,
    np.array([np.random.rand(n)*10+5,np.random.rand(n)*10-10,np.random.rand(n)*10+5]).T,
    np.array([np.random.rand(n)*15-15,np.random.rand(n)*15,np.random.rand(n)*15+15]).T
]
mean = np.concatenate(data,axis=0).mean(axis=0)

axes = NPLDA.binary_relevance_axes(data)

projections = [[np.dot(samples,axes[i]) for i in range(len(colours))] for samples in data]

fig = plt.figure()
scale = 20
ax = fig.add_subplot(projection='3d')
plt.title("One-versus-all Discriminant Axes for 4 classes in 3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
for i in range(len(colours)):
    ax.scatter(data[i][:,0],data[i][:,1],data[i][:,2],c=colours[i])
    ax.plot([mean[0]-axes[i][0]*scale,mean[0]+axes[i][0]*scale],[mean[1]-axes[i][1]*scale,mean[1]+axes[i][1]*scale],[mean[2]-axes[i][2]*scale,mean[2]+axes[i][2]*scale],c=colours[i])
fig,axs = plt.subplots(2,2)
for i in range(len(colours)):
    axs[i%2,math.floor(i/2)].set_yticks([1,2,3,4])
    axs[i%2,math.floor(i/2)].set_ylabel("Class")
    axs[i%2,math.floor(i/2)].set_xlabel("Projection")
    axs[i%2,math.floor(i/2)].set_title(f"{names[i]} vs. others")
    for j in range(len(colours)):
        #axs[i%2,math.floor(i/2)].scatter(projections[j][i],np.zeros(projections[j][i].shape[0]),c=colours[j],s=1)
        axs[i%2,math.floor(i/2)].scatter(projections[j][i],(j+1)*np.ones(projections[j][i].shape[0]),c=colours[j],s=1)
fig.tight_layout()
plt.show()