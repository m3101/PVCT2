import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

indices = list(range(15))+[17,18]
positions = ["neck","chest","lelbow","lhand","lfingers","relbow","rhand",
             "rfingers","hip","lhip","lknee","lfoot","rhip","rknee","rfoot","lshoulder","rshoulder"]
connections = [
    ["neck","chest"],
    ["chest","lshoulder"],
    ["lshoulder","lelbow"],
    ["lelbow","lhand"],
    ["chest","rshoulder"],
    ["rshoulder","relbow"],
    ["relbow","rhand"],
    ["chest","hip"],
    ["lhip","lknee"],
    ["lknee","lfoot"],
    ["rhip","rknee"],
    ["rknee","rfoot"],
    ["hip","rhip"],
    ["hip","lhip"]
]
with open("numpyData/person14a.np",'rb') as i_f:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    skel = np.load(i_f)[0][indices]
    skel = skel-skel.mean(axis=0)
    i=11
    idx = np.arange(skel.shape[0])
    ax.scatter(skel[idx!=i,0],skel[idx!=i,1],skel[idx!=i,2],c="b")
    ax.scatter(skel[i,0],skel[i,1],skel[i,2],c="k")
    for conn in connections:
        ax.plot(skel[[positions.index(conn[0]),positions.index(conn[1])],0],
                skel[[positions.index(conn[0]),positions.index(conn[1])],1],
                skel[[positions.index(conn[0]),positions.index(conn[1])],2],
                c="b")
plt.show()
#exit(0)
names = [f"person2{i}" for i in range(1,3)]+[f"person1{i}" for i in range(1,7)]+[f"person3{i}" for i in range(1,7)]+[f"person4{i}" for i in range(1,7)]
for name in names:
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.set_xlabel("X")
    #ax.set_ylabel("Y")
    #plt.title(f"{name}")
    fig,axs = plt.subplots(2)
    axs[0].set_title(f"Keyframes for {name}")
    axs[0].set_xlabel("")
    axs[0].set_xticks([])
    axs[0].set_ylabel("")
    axs[0].set_yticks([])
    axs[1].set_xlabel("")
    axs[1].set_xticks([])
    axs[1].set_ylabel("")
    axs[1].set_yticks([])
    for f,c,i in zip("abcd","rgbk",[0,1,2,3]):
        with open(f"numpyData/{name}{f}.np","rb") as i_f:
            skel = np.load(i_f)[0][:len(positions)]
            skel = skel-np.repeat(skel.mean(axis=0)[np.newaxis,:],skel.shape[0],axis=0)
            #3d plot
            #ax.scatter(skel[:,0],skel[:,1],skel[:,2]+i*0.1*np.ones(skel.shape[0]),c=c)
            #for conn in connections:
                #ax.plot(skel[[positions.index(conn[0]),positions.index(conn[1])],0],
                        #skel[[positions.index(conn[0]),positions.index(conn[1])],1],
                        #skel[[positions.index(conn[0]),positions.index(conn[1])],2]+
                        #i*0.1*np.ones(2),
                        #c=c)
            #2d plots
            #ZY plane
            s1=70
            sk2d = np.dot(skel,np.array([[-1,0],[0,-1],[0,0]]))
            axs[0].scatter(sk2d[:,0]+s1*i*np.ones(sk2d.shape[0]),sk2d[:,1],c=c)
            for conn in connections:
                axs[0].plot(sk2d[[positions.index(conn[0]),positions.index(conn[1])],0]+s1*i*np.ones(2),
                        sk2d[[positions.index(conn[0]),positions.index(conn[1])],1],
                        c=c)
            s2=0.2
            sk2d = np.dot(skel,np.array([[0,0],[0,-1],[-1,0]]))
            axs[1].scatter(sk2d[:,0]+s2*i*np.ones(sk2d.shape[0]),sk2d[:,1],c=c)
            for conn in connections:
                axs[1].plot(sk2d[[positions.index(conn[0]),positions.index(conn[1])],0]+s2*i*np.ones(2),
                        sk2d[[positions.index(conn[0]),positions.index(conn[1])],1],
                        c=c)
            fig.tight_layout()
plt.show()