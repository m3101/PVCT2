"""
    LDA for Machine learning toolset using Numpy functions
    Copyright (C) 2021 Amélia O. F. da S.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

class SingularError(Exception):
    def __str__(self):
        return "LDA is not possible for a number of samples smaller than the number of dimensions for each class."

def LDA(data:list)->np.ndarray:
    """
    Calculates the LDA from a list of (samples,dimensions) numpy arrays.
    * List = [class1, class2, class3, ...]
    * With each class being [sample1, sample2, sample3, ...]
    * And each sample being [dimension1,dimension2, ...]
    """

    #Width of each data matrix (length of each sample vector)
    n = data[0].shape[1]
    #Smallest number of samples
    mins = min(map(len,data))
    if mins<n:
        raise SingularError()

    #Number of classes
    m = len(data)

    totalmean = sum([c.sum(axis=0) for c in data])/sum([len(c) for c in data])
    class_means = [c.mean(axis=0) for c in data]

    #Each class' distance from the mean
    class_distances = [(class_means[i]-totalmean) for i in range(m)]

    #Scatter between each class mean and the absolute mean
    scatter_total = sum([len(c)*np.dot(class_distances[i][:,np.newaxis],class_distances[i][:,np.newaxis].T) for i,c in enumerate(data)])
    #Intrenal scatter for each class
    scatter_internal = sum([sum([np.dot((vector-class_means[i])[:,np.newaxis],(vector-class_means[i])[:,np.newaxis].T) for vector in c]) for i,c in enumerate(data)])
    vt,_,vt = np.linalg.svd(np.dot(scatter_total,np.linalg.inv(scatter_internal)))
    return vt

def binary_relevance_axes(data:list)->np.ndarray:
    """
    Calculates the one-versus-all Fisher Discriminant axes for classifying within a
    list of (samples,dimensions) arrays with the following format:
    * [class1, class2, class3, ...]
    * With each class being [sample1, sample2, sample3, ...]
    * And each sample being [dimension1,dimension2, ...]
    Returns the projection vectors
    """
    return np.array([LDA([
                            data[i],
                            np.concatenate(data[:i]+data[i+1:],axis=0)#-
                                #np.repeat(mean[np.newaxis,:],sum(map(len,data[:i]))+sum(map(len,data[i+1:])),axis=0)
                        ])[0]
                    for i in range(len(data))])

def project_on_axes(data:list,axes:tuple)->list:
    """
    Projects the data onto the axes specified by a (vects,mean) tuple.
    """
    vects,mean = axes
    return [
        np.dot((class_samples-np.repeat(mean[np.newaxis,:],class_samples.shape[0],axis=0)),vects.T)
        for class_samples in data
    ]