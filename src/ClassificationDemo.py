import NPLDA
import numpy as np
import matplotlib.pyplot as plt

n = 100
names = ['r','b','g']
training = [
    np.array([np.random.rand(n)*10-5,np.random.rand(n)*10+10]).T,
    np.array([np.random.rand(n)*10-15,np.random.rand(n)*10-10]).T,
    np.array([np.random.rand(n)*10+5,np.random.rand(n)*10-10]).T
]
n=1000
classification = np.array([np.random.rand(n)*20-10,np.random.rand(n)*20-5]).T

vec = NPLDA.binary_relevance_axes(training)
classes = np.array([names[np.argmin([np.linalg.norm(sample-reference) for reference in [np.mean(class_samples, axis=0) for class_samples in training]])] for sample in classification])

plt.title("Binary relevance LDA axes for three classes in 2d")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(training[0][:,0],training[0][:,1],c='r')
plt.scatter(training[1][:,0],training[1][:,1],c='b')
plt.scatter(training[2][:,0],training[2][:,1],c='g')
centre = (0,0)
plt.plot([centre[0],centre[0]+vec[0][0]*10],[centre[1],centre[1]+vec[0][1]*10],c='r')
plt.plot([centre[0],centre[0]+vec[1][0]*10],[centre[1],centre[1]+vec[1][1]*10],c='b')
plt.plot([centre[0],centre[0]+vec[2][0]*10],[centre[1],centre[1]+vec[2][1]*10],c='g')
plt.figure()
plt.title("Classification by nearest centroid")
plt.xlabel("X")
plt.ylabel("Y")
for c in names:
    if len(classification[classes==c])>0:
        plt.scatter(classification[classes==c][:,0],classification[classes==c][:,1],c=c)

references = [np.dot(training[i],vector).mean(axis=0) for i,vector in enumerate(vec)]
classes = np.array([names[np.argmin([np.abs(np.dot(classification[i],vector)-references[j]) for j,vector in enumerate(vec)])] for i in range(classification.shape[0])])

plt.figure()
plt.title("Classification by testing individual axes")
plt.xlabel("X")
plt.ylabel("Y")
for c in names:
    if len(classification[classes==c])>0:
        plt.scatter(classification[classes==c][:,0],classification[classes==c][:,1],c=c)
plt.show()