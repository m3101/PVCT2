import numpy as np
import os
import re
from matplotlib import pyplot as plt
import seaborn as sns
import math
import pandas as pd

#Loading extracted data
samples = {name[1] for name in {re.search("(person.)\.np",f) for f in os.listdir("classification")} if name}
samples = {name:np.load(open("classification/"+name+".np","rb")) for name in samples}

with open("classification/labels.np","rb") as i_f:
    labels = np.load(i_f)
with open("classification/references.np","rb") as i_f:
    references = np.load(i_f)
with open("classification/means.np","rb") as i_f:
    means = np.load(i_f)
with open("classification/thresholds.np","rb") as i_f:
    thresholds = np.load(i_f)
with open("classification/projectionMatrix.np","rb") as i_f:
    projectionMatrix = np.load(i_f)

#Translating the means to our new origin
means = means - np.repeat(references[np.newaxis,:],means.shape[0],axis=0)

#Projecting onto the projection axes and subtracting the mean
samples = {name:np.dot(samples[name],projectionMatrix) for name in samples}
samples = {name:samples[name]-np.repeat(references[np.newaxis,:],samples[name].shape[0],axis=0) for name in samples}
samples = {n:samples[n] for n in sorted(samples)}

fig,axs = plt.subplots(2,math.ceil(len(samples)/2))
plt.title("Projections along the one-against-all discriminant axes")

colours = ['r','g','b','purple']

for i in range(len(samples)):
    axs[i%2,math.floor(i/2)].set_yticks(list(range(1,len(samples)+1)))
    axs[i%2,math.floor(i/2)].set_ylabel("Person")
    axs[i%2,math.floor(i/2)].set_xlabel("Projection")
    axs[i%2,math.floor(i/2)].set_title(f"{labels[i]} vs. others")
    for j,key in enumerate(samples):
        axs[i%2,math.floor(i/2)].scatter(samples[key][:,i],(j+1)*np.ones(samples[key].shape[0]),c=colours[j],s=2)
plt.tight_layout()

fig,axs = plt.subplots(2,math.ceil(len(samples)/2))
plt.title("Projections along the one-against-all discriminant axes")

colours = ['r','g','b','purple']

for i in range(len(samples)):
    axs[i%2,math.floor(i/2)].set_yticks(list(range(1,len(samples)+1)))
    axs[i%2,math.floor(i/2)].set_ylabel("Person")
    axs[i%2,math.floor(i/2)].set_xlabel("Projection")
    axs[i%2,math.floor(i/2)].set_title(f"{labels[i]} vs. others")
    for j,key in enumerate(samples):
        axs[i%2,math.floor(i/2)].scatter(samples[key][:,i],(j+1)*np.ones(samples[key].shape[0]),c=colours[j],s=2)
        axs[i%2,math.floor(i/2)].axvline(thresholds[i],c=colours[i])
        axs[i%2,math.floor(i/2)].axvline(-thresholds[i],c=colours[i])
plt.tight_layout()
plt.figure()

#Classification using nearest mean
confusion_matrix = {original:{predicted:0 for predicted in labels} for original in labels}
for i,original in enumerate(labels):
    for sample in samples[original]:
        predicted = labels[np.argmin((((means-np.repeat(sample[np.newaxis,:],means.shape[0],axis=0))**2).sum(axis=1)))]
        confusion_matrix[original][predicted]+=1
sns.heatmap(pd.DataFrame.from_dict(confusion_matrix),annot=True,cmap="hot")

#Classification using thresholds and a single projection
#a.k.a. "finding a person"
confusion_matrix = {
    "Positive":{
        "Positive":0,
        "Negative":0
    },
    "Negative":{
        "Positive":0,
        "Negative":0
    }
}
plt.figure()
for i,original in enumerate(labels):
    for j,c in enumerate(labels):
        for sample in samples[c]:
            predicted = abs(sample[i])<thresholds[i]
            confusion_matrix["Positive" if j==i else "Negative"]["Positive" if predicted else "Negative"]+=1
sns.heatmap(pd.DataFrame.from_dict(confusion_matrix),annot=True,cmap="hot")

plt.show()