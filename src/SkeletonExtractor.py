import sys
#Follow the installation instructions at the docs
sys.path.append('/usr/local/python');
from openpose import pyopenpose as op
import cv2
import numpy as np
import os
import re

opWrapper = op.WrapperPython()
params = {}

#This should point to the cloned repository's "models" directory.
params["model_folder"] = "/home/m31/Tools/openpose/models"
opWrapper.configure(params)
opWrapper.start()

files = os.listdir("trainingData")
#Finds all person[X][S][abcd].png samples in the trainingData directory
names = [[
            sorted([final[1] for final in [re.search("("+f[0]+".+)\.png",n2) for n2 in files] if final])
            for f in {re.search(nome[1]+".",n) for n in files} if f
         ]
         for nome in {re.search(r"(person.)", nome)
                      for nome in files}
         if nome]
names = np.array(names).flatten()
for name in names:
    print(f"Processing image {name}")
    try:
        open(f"numpyData/{name}.np",'rb').close()
    except Exception as e:
        print(e)
        datum = op.Datum()
        img = cv2.imread(f"trainingData/{name}.png")
        if img is None:
            print("What in hell is happening?")
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print(f"Body Keypoints: {datum.poseKeypoints}")
        with open(f"numpyData/{name}.np","wb") as o_f:
            np.save(o_f,np.array(datum.poseKeypoints))