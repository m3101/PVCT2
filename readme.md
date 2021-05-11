# Project 2 - Statistical gait analysis

## How to use

Each file in the `src` generates a series of matrices or images for the final documents.

Specific instructions are available on the "File Structure" section below.

The execution order for reproducing the results shown is

* SkeletonVis.py
* 3dLDADemo.py
* OldLDADemo.py
* OtherIllustrations.py
* FeatureExtract.py
* FeatureVis.py

## File structure
```.
├── classification
│   - Contains the files used in 
│     the classification demos
├── numpyData
│   - Contains the extracted pose
│     vectors
├── readme.md
├── relatorio
│   - Contains the files used for
│     generating the final documents
├── src
│   ├── 3dLDADemo.py
│   │   - Generates 3D LDA projections
│   │     (for illustration purposes)
│   ├── FeatureExtract.py
│   │   - Extracts the projection matrices
│   │     from pose data.
│   │     (assumes the numpyData directory
│   │      exists and contains .np files)
│   ├── FeatureVis.py
│   │   - Generates the projection images
│   ├── LICENSE
│   ├── NPLDA.py
│   │   - LDA implementation
│   ├── OldLDADemo.py
│   │   - Generates 2D LDA projections
│   ├── OtherIllustrations.py
│   │   - Generates secondary illustrations
│   ├── README.md
│   ├── SkeletonExtractor.py
│   │   - Extracts pose data from images
│   │     (assumes the trainingData
│   │      directory exists and contains
│   │      groups of 4 png images)
│   └── SkeletonVis.py
        - Generates the pose projection
          images
```