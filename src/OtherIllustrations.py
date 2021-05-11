import numpy as np
from matplotlib import pyplot as plt
np.random.seed(42)

xs = np.random.randn(100)
ys = np.random.randn(100)+xs

svd,_,_ = np.linalg.svd(np.concatenate([xs[np.newaxis,:],ys[np.newaxis,:]],axis=0))
svd = svd*5

plt.scatter(xs,ys,c="k")
plt.plot([-svd[0,0],svd[0,0]],[-svd[0,1],svd[0,1]],c="r")
plt.show()

plt.figure()

xs = np.random.randn(100)
ys = np.random.randn(100)*.1
m1 = np.array((xs.mean(),ys.mean()))
plt.scatter(xs,ys,c="b")
rxs = np.random.randn(100)*3-4
rys = np.random.randn(100)*.1+.5
plt.scatter(rxs,rys,c="r")
m2 = np.array((rxs.mean(),rys.mean()))
px = np.concatenate([xs,rxs])
py = np.concatenate([ys,rys])
labels =(((np.c_[px,py]-np.repeat(m1[np.newaxis,:],px.shape[0],axis=0))**2).sum(axis=1)<
         ((np.c_[px,py]-np.repeat(m2[np.newaxis,:],px.shape[0],axis=0))**2).sum(axis=1))
plt.figure()
plt.scatter(px[labels],py[labels],c='b')
plt.scatter(px[~labels],py[~labels],c='r')
plt.scatter(m1[0],m1[1],c='k')
plt.scatter(m2[0],m2[1],c='k')
plt.show()