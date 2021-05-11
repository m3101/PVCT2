import NPLDA
import numpy as np
import matplotlib.pyplot as plt

n = 100
data = [
    np.array([np.random.rand(n)*10-5,np.random.rand(n)*10+10]).T,
    np.array([np.random.rand(n)*10-15,np.random.rand(n)*10-10]).T,
    np.array([np.random.rand(n)*10+5,np.random.rand(n)*10-10]).T
]

vec = NPLDA.binary_relevance_axes(data[:2])
print(vec)
plt.title("Fisher's Linear Discriminant Axis")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(data[0][:,0],data[0][:,1],c='r')
plt.scatter(data[1][:,0],data[1][:,1],c='b')
centre = (-5,5)
plt.plot([centre[0]-vec[0]*20,centre[0]+vec[0]*20],[centre[1]-vec[1]*20,centre[1]+vec[1]*20],color="black")

plt.figure()
plt.title("Projection onto the discriminant axis")
normsq=np.dot(vec[0],vec[0])
projection = [np.array([vec[0]*np.dot(vec[0],p)/normsq for p in c]) for c in data]
plt.xlabel("LDA[0]")
plt.ylabel("Class")
plt.yticks([1,2])
plt.scatter(projection[0][:,0],1*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],2*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[0][:,0],0*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],0*np.ones((len(projection[0]))),c='b')
plt.show()

vec = NPLDA.binary_relevance_axes(data)
print(vec)
plt.title("Binary relevance LDA axes for three classes in 2d")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(data[0][:,0],data[0][:,1],c='r')
plt.scatter(data[1][:,0],data[1][:,1],c='b')
plt.scatter(data[2][:,0],data[2][:,1],c='g')
centre = (0,0)
plt.plot([centre[0],centre[0]+vec[0][0]*10],[centre[1],centre[1]+vec[0][1]*10],c='r')
plt.plot([centre[0],centre[0]+vec[1][0]*10],[centre[1],centre[1]+vec[1][1]*10],c='b')
plt.plot([centre[0],centre[0]+vec[2][0]*10],[centre[1],centre[1]+vec[2][1]*10],c='g')

plt.figure()
plt.title("Projection on the first axis (red vs. all)")
normsq=np.dot(vec[0],vec[0])
projection = [np.array([vec[0]*np.dot(vec[0],p)/normsq for p in c]) for c in data]
plt.xlabel("LDA[0]")
plt.ylabel("Class")
plt.yticks([1,2,3])
plt.scatter(projection[0][:,0],1*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],2*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],3*np.ones((len(projection[0]))),c='g')

plt.scatter(projection[0][:,0],0*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],0*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],0*np.ones((len(projection[0]))),c='g')
plt.figure()
plt.title("Projection on the second axis (blue vs. all)")
normsq=np.dot(vec[1],vec[1])
projection = [np.array([vec[1]*np.dot(vec[1],p)/normsq for p in c]) for c in data]
plt.xlabel("LDA[1]")
plt.ylabel("Class")
plt.yticks([1,2,3])
plt.scatter(projection[0][:,0],1*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],2*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],3*np.ones((len(projection[0]))),c='g')
plt.scatter(projection[0][:,0],0*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],0*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],0*np.ones((len(projection[0]))),c='g')
plt.figure()
plt.title("Projection on the third axis (green vs. all)")
normsq=np.dot(vec[2],vec[2])
projection = [np.array([vec[2]*np.dot(vec[2],p)/normsq for p in c]) for c in data]
plt.xlabel("LDA[2]")
plt.ylabel("Class")
plt.yticks([1,2,3])
plt.scatter(projection[0][:,0],1*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],2*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],3*np.ones((len(projection[0]))),c='g')
plt.scatter(projection[0][:,0],0*np.ones((len(projection[0]))),c='r')
plt.scatter(projection[1][:,0],0*np.ones((len(projection[0]))),c='b')
plt.scatter(projection[2][:,0],0*np.ones((len(projection[0]))),c='g')
plt.show()