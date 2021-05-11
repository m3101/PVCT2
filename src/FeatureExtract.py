import numpy as np
import math
import NPLDA
import os
import re
from matplotlib import pyplot as plt

indices = list(range(15))+[17,18]
positions = ["neck","chest","lelbow","lhand","lfingers","relbow","rhand",
             "rfingers","hip","lhip","lknee","lfoot","rhip","rknee","rfoot","lshoulder","rshoulder"]
angnames = list(np.concatenate([[f"Left elbow (moment {i})",
             f"Right elbow (moment {i})",
             f"Left armpit (moment {i})",
             f"Right armpit (moment {i})",
             f"Left knee (moment {i})",
             f"Right knee (moment {i})",
             f"Left hip (moment {i})",
             f"Right hip (moment {i})"] for i in range(4)],axis=0))
featnames = angnames+[f"Left elbow (Standard Deviation)",
             f"Right elbow (Standard Deviation)",
             f"Left armpit (Standard Deviation)",
             f"Right armpit (Standard Deviation)",
             f"Left knee (Standard Deviation)",
             f"Right knee (Standard Deviation)",
             f"Left hip (Standard Deviation)",
             f"Right hip (Standard Deviation)"]
angref = [
    ["lshoulder","lelbow","lhand"],#Left elbow angle
    ["rshoulder","relbow","rhand"],#Right elbow angle
    ["chest","lshoulder","lelbow"],#Left armpit angle
    ["chest","rshoulder","relbow"],#Right armpit angle
    ["lhip","lknee","lfoot"],#Left knee angle
    ["rhip","rknee","rfoot"],#Right knee angle
    ["hip","lhip","lknee"],#Left hip angle
    ["hip","rhip","rknee"]#Right hip angle
]
names = sorted({nome[1] for nome in [re.search(r"(person.)", nome) for nome in os.listdir("numpyData")] if nome})

def angle(points):
    v1 = points[1]-points[2]
    v2 = points[1]-points[0]
    p1 = np.dot(v1,v2)/np.dot(v2,v2)
    p2 = np.dot(v1,v2)/np.dot(v1,v1)
    return math.acos(p2 if abs(p1)>abs(p2) else p1)
files = os.listdir("numpyData")
samples = {n:[] for n in names}

#Finds all person[X][S] samples in the numpyData directory
names = sorted({nome[1]
         for nome in {re.search(r"(person..)", nome)
                      for nome in files}
         if nome})
#Calculates all the angle and deviation features
for name in names:
    angles = []
    for f,c,i in zip("abcd","rgbk",[0,1,2,3]):
        with open(f"numpyData/{name}{f}.np","rb") as i_f:
            skel = np.load(i_f)[0][:len(positions)]
            skel = skel-np.repeat(skel.mean(axis=0)[np.newaxis,:],skel.shape[0],axis=0)
            angles = angles + [angle(skel[[positions.index(n) for n in a]])/np.pi for a in angref]
    stdev = [np.std(angles[i::8]) for i in range(8)]
    for c in samples:
        if c in name:
            samples[c].append(angles+stdev)
for c in samples:
    with open(f"classification/{c}.np","wb") as o_f:
        np.save(o_f,samples[c])

#SVD Dimensionality reduction to fit the LDA (at least as little dimensions as the
# number of samples of the class with the least samples)
_,_,vt = np.linalg.svd(np.concatenate(list(samples.values()),axis=0))
svd = np.array(vt[:len(vt) if len(vt)<min([len(samples[s]) for s in samples]) else min([len(samples[s]) for s in samples])]).T
projected = {c:np.dot(samples[c],svd) for c in samples}
#Class-wise LDA
lda = NPLDA.binary_relevance_axes([projected[c] for c in projected])
lda = lda.T

#Final projection
projection = np.dot(svd,lda)
for i in range(projection.shape[1]):
    fig,ax = plt.subplots()
    weights = np.abs(projection[:,i])/np.abs(projection[:,i]).sum()
    pairs = sorted(zip(100*weights,featnames),key=lambda a:a[0],reverse=True)[:5]
    print(pairs)
    fnames = [p[1] for p in pairs]
    values = [p[0] for p in pairs]
    ys = np.arange(5)
    ax.barh(ys,values)
    ax.set_yticks(ys)
    ax.set_yticklabels(fnames)
    ax.set_xlabel("Projection weight (%)")
    ax.set_title(f"Person {i+1}")
    ax.invert_yaxis()
    plt.tight_layout()
plt.show()

#Means for each class in their own axis
references = [np.dot(samples[c],projection).mean(axis=0)[i] for i,c in enumerate(projected)]

#Means for the all classes on all axes
means = np.array([np.dot(samples[c],projection).mean(axis=0) for i,c in enumerate(projected)])

#Midpoints between the reference and the minimum of the other classes' means on that axis
thresholds = [np.abs(means[:,i][means[:,i]!=references[i]]-references[i]).min()/2 for i in range(len(means))]
labels = [c for c in projected]
with open("classification/projectionMatrix.np","wb") as o_f:
    np.save(o_f,projection)
with open("classification/means.np","wb") as o_f:
    np.save(o_f,np.array(means))
with open("classification/references.np","wb") as o_f:
    np.save(o_f,np.array(references))
with open("classification/labels.np","wb") as o_f:
    np.save(o_f,labels)
with open("classification/thresholds.np","wb") as o_f:
    np.save(o_f,thresholds)