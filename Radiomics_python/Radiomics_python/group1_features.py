import numpy as np
import math
#import scipy.stats as stat

import profiling_tools

"""
Group 1 : First order statistics
X : numpy 1-d array

last update : 2016.12.26 by K.Kobayashi
"""

def group1_features(image):

    X = image.ravel()

    group1_features = {}

    group1_features["energy"] = energy(X)
    group1_features["entropy"] = entropy(X)
    group1_features["kurtosis"] = kurtosis(X)
    group1_features["maximum"] = maximum(X)
    group1_features["mean"] = mean(X)
    group1_features["mean absolute deviation"] = mean_absolute_deviation(X)
    group1_features["median"] = median(X)
    group1_features["minimum"] = minimum(X)
    group1_features["range"] = _range(X)
    group1_features["root mean square"] = root_mean_square(X)
    group1_features["skewness"] = skewness(X)
    group1_features["standard deviation"] = standard_deviation(X)
    group1_features["uniformity"] = uniformity(X)
    group1_features["variance"] = variance(X)

    print group1_features

    return group1_features

def energy(X):

    return np.dot(X, X)

def entropy(X):

    hist, bin_edges = np.histogram(X, density = True)

    return - np.sum(hist * np.ma.log2(hist))

def kurtosis(X):

    return np.mean((X-X.mean())**4) / (np.std(X)**4) # stat.kurtosis(X, fisher = False)

def maximum(X):

    return np.max(X)

def mean(X):

    return np.mean(X)

def mean_absolute_deviation(X):

    return np.mean(np.abs(X - X.mean()))

def median(X):

    return np.median(X)

def minimum(X):

    return np.min(X)

def _range(X):

    return np.max(X) - np.min(X)

def root_mean_square(X):

    return math.sqrt(energy(X)/len(X))

def skewness(X):

    return np.mean((X-X.mean())**3) / (np.std(X)**3)

def standard_deviation(X):

    return np.std(X, ddof=1) # math.sqrt(np.sum((X-X.mean())**2)/(len(X)-1))

def uniformity(X):

    hist, bin_edges = np.histogram(X, density = True)

    return np.dot(hist, hist)

def variance(X):

    return np.var(X, ddof=1) # np.sum((X-X.mean())**2)/(len(X)-1)