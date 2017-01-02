import pyximport
import numpy as np
import math

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glcm_loop import _3d_glcm_vector_loop 

import profiling_tools

"""
Group 3 : Textural features (Class version)

Updated 2017/1/2 Kaz-K
"""

def group3_glcm_features(image, distance):

    autocorrelation = []
    cluster_prominence = []
    cluster_shade = []
    cluster_tendency = []
    contrast = []
    correlation = []
    difference_entropy = []
    dissimilarity = []
    energy = []
    entropy = []
    homogeneity_1 = []
    homogeneity_2 = []
    IMC_1 = []
    IMC_2 = []
    IDMN = []
    IDN = []
    inverse_variance = []
    maximum_probability = []
    sum_average = []
    sum_entropy = []
    sum_variance = []
    variance = []

    vectors = [[0, 1, 1],[-1, 1, 1],[-1, 0, 1], 
               [-1, -1, 1],[0, -1, 1],[1, -1, 1], 
               [1, 0, 1],[1, 1, 1],[0, 0, 1],
               [0, 1, 0],[-1, 1, 0],[-1, 0, 0],[-1, -1, 0]]

    glcm_features = {}

    for vector in vectors:

        glcm_matrix = GLCM_Matrix(image, distance, vector)

        autocorrelation.append(glcm_matrix.autocorrelation())
        cluster_prominence.append(glcm_matrix.cluster_prominence())
        cluster_shade.append(glcm_matrix.cluster_shade())
        cluster_tendency.append(glcm_matrix.cluster_tendency())
        contrast.append(glcm_matrix.contrast())
        correlation.append(glcm_matrix.correlation())
        difference_entropy.append(glcm_matrix.difference_entropy())
        dissimilarity.append(glcm_matrix.dissimilarity())
        energy.append(glcm_matrix.energy())
        entropy.append(glcm_matrix.entropy())
        homogeneity_1.append(glcm_matrix.homogeneity1())
        homogeneity_2.append(glcm_matrix.homogeneity2())
        IMC_1.append(glcm_matrix.IMC1())
        IMC_2.append(glcm_matrix.IMC2())
        IDMN.append(glcm_matrix.IDMN())
        IDN.append(glcm_matrix.IDN())
        inverse_variance.append(glcm_matrix.inverse_variance())
        maximum_probability.append(glcm_matrix.maximum_probability())
        sum_average.append(glcm_matrix.sum_average())
        sum_entropy.append(glcm_matrix.sum_entropy())
        sum_variance.append(glcm_matrix.sum_variance())
        variance.append(glcm_matrix.variance())

    glcm_features['autocorrelation'] = np.mean(autocorrelation)
    glcm_features['cluster_prominence'] = np.mean(cluster_prominence)
    glcm_features['cluster_shade'] = np.mean(cluster_shade)
    glcm_features['cluster_tendency'] = np.mean(cluster_tendency)
    glcm_features['contrast'] = np.mean(contrast)
    glcm_features['correlation'] = np.mean(correlation)
    glcm_features['difference_entropy'] = np.mean(difference_entropy)
    glcm_features['dissimilarity'] = np.mean(dissimilarity)
    glcm_features['energy'] = np.mean(energy)
    glcm_features['entropy'] = np.mean(entropy)
    glcm_features['homogeneity_1'] = np.mean(homogeneity_1)
    glcm_features['homogeneity_2'] = np.mean(homogeneity_2)
    glcm_features['IMC_1'] = np.mean(IMC_1)
    glcm_features['IMC_2'] = np.mean(IMC_2)
    glcm_features['IDMN'] = np.mean(IDMN)
    glcm_features['IDN'] = np.mean(IDN)
    glcm_features['inverse_variance'] = np.mean(inverse_variance)
    glcm_features['maximum_probability'] = np.mean(maximum_probability)
    glcm_features['sum_average'] = np.mean(sum_average)
    glcm_features['sum_entropy'] = np.mean(sum_entropy)
    glcm_features['sum_variance'] = np.mean(sum_variance)
    glcm_features['variance'] = np.mean(variance)

    print glcm_features

    return glcm_features

class GLCM_Matrix:

    def __init__(self, image):

        self.image = image.astype(np.uint16)
        self.glcm_matrix = None
        self.levels = None
        self.i = None
        self.j = None

        self.px = None
        self.py = None
        self.ux = None
        self.uy = None
        self.sigx = None
        self.sigy = None

        self.diff = None
        self.sum = None

        self.HX = None
        self.HY = None
        self.HXY = None
        self.HXY1 = None
        self.HXY2 = None

    def calc_GLCM(self, distance, vector):

        glcm_matrix = _3d_glcm_vector_loop(self.image, distance, vector[0], vector[1], vector[2])

        # normalize glcm_matrix to be probabilities

        self.glcm_matrix = glcm_matrix.astype(np.float) / np.sum(glcm_matrix)

        self.levels = np.arange(1, self.glcm_matrix.shape[0]+1).astype(np.float)

        i, j = np.meshgrid(self.levels, self.levels)

        self.i = i.T
        self.j = j.T

        self.px = np.sum(self.glcm_matrix, axis = 1) #px is the marginal row probabilities
        self.py = np.sum(self.glcm_matrix, axis = 0) #py is the marginal column probabilities

        self.ux = np.sum(self.i * self.glcm_matrix) #np.sum(self.levels * self.px)
        self.uy = np.sum(self.j * self.glcm_matrix) #np.sum(self.levels * self.py)

        self.sigx = np.sum((self.i - self.ux)**2 * self.glcm_matrix)
        self.sigy = np.sum((self.j - self.uy)**2 * self.glcm_matrix)

        _diff = np.zeros(len(self.levels))

        for i, val in enumerate(np.abs(self.i - self.j).ravel()):

            _diff[int(val)] += self.glcm_matrix.ravel()[i]

        self.diff = _diff

        _sum = np.zeros(2 * len(self.levels) - 1)

        for i, val in enumerate((self.i + self.j).ravel()):

            _sum[int(val)-2] += self.glcm_matrix.ravel()[i]

        self.sum = _sum

        self.HX = - np.sum(self.px * np.ma.log2(self.px))
        self.HY = - np.sum(self.py * np.ma.log2(self.py))
        self.HXY = - np.sum(self.glcm_matrix * np.ma.log2(self.glcm_matrix))

        _px, _py = np.meshgrid(self.px, self.py)
        
        _px = _px.T
        _py = _py.T

        self.HXY1 = - np.sum(self.glcm_matrix * np.ma.log2(_px * _py))
        self.HXY2 = - np.sum(_px * _py * np.ma.log2(_px * _py))

        print "autocorrelation: ", self.autocorrelation()
        print "cluster prominence: ", self.cluster_prominence()
        print "cluster shade: ", self.cluster_shade()
        print "cluster tendency: ", self.cluster_tendency()
        print "contrast: ", self.contrast()
        print "correlation: ", self.correlation()
        print "difference entropy: ", self.difference_entropy()
        print "dissimilarity: ", self.dissimilarity()
        print "energy: ", self.energy()
        print "entropy: ", self.entropy()
        print "homogeneity 1: ", self.homogeneity1()
        print "homogeneity 2: ", self.homogeneity2()
        print "IMC 1:, ", self.IMC1()
        print "IMC 2:, ", self.IMC2()
        print "IDMN: ", self.IDMN()
        print "IDN: ", self.IDN()
        print "inverse variance: ", self.inverse_variance()
        print "maximum probability: ", self.maximum_probability()
        print "sum average: ", self.sum_average()
        print "sum entropy: ", self.sum_entropy()
        print "sum variance: ", self.sum_variance()
        print "variance: ", self.variance()

    def autocorrelation(self):

        return np.sum(self.i * self.j * self.glcm_matrix)

    def cluster_prominence(self):

        return np.sum((self.i + self.j - self.ux - self.uy)**4 * self.glcm_matrix)

    def cluster_shade(self):

        return np.sum((self.i + self.j - self.ux - self.uy)**3 * self.glcm_matrix)

    def cluster_tendency(self):

        return np.sum((self.i + self.j - self.ux - self.uy)**2 * self.glcm_matrix)

    def contrast(self):

        return np.sum((self.i - self.j)**2 * self.glcm_matrix)

    def correlation(self):

        return (np.sum(self.i * self.j * self.glcm_matrix) - self.ux * self.uy) / (self.sigx * self.sigy)

    def difference_entropy(self):

        return - np.sum(self.diff * np.ma.log2(self.diff))

    def dissimilarity(self):

        return np.sum(np.abs(self.i - self.j) * self.glcm_matrix)

    def energy(self):

        return np.sum(self.glcm_matrix**2)

    def entropy(self):

        return self.HXY

    def homogeneity1(self):

        return np.sum((self.glcm_matrix) / (1 + np.abs(self.i - self.j)))

    def homogeneity2(self):

        return np.sum((self.glcm_matrix) / (1 + np.abs(self.i - self.j)**2))

    def IMC1(self):

        return (self.HXY - self.HXY1) / max(self.HX, self.HY)

    def IMC2(self):

        return math.sqrt(1.0 - math.exp(-2.0 * (self.HXY2 - self.HXY)))

    def IDMN(self):

        return np.sum(self.glcm_matrix / (1.0 + (self.i - self.j)**2 / len(self.levels)**2))

    def IDN(self):

        return np.sum(self.glcm_matrix / (1.0 + np.abs(self.i - self.j) / len(self.levels)))

    def inverse_variance(self):

        _mask = np.ma.masked_less_equal(np.abs(self.i - self.j), 0).mask

        return np.sum(np.ma.array(self.glcm_matrix, mask = _mask) / np.ma.array(np.abs(self.i - self.j)**2, mask = _mask))

    def maximum_probability(self):

        return np.max(self.glcm_matrix)

    def sum_average(self):

        return np.sum(np.arange(2, 2 * len(self.levels) + 1) * self.sum)

    def sum_entropy(self):

        return - np.sum(self.sum * np.ma.log2(self.sum))

    def sum_variance(self):

        return np.sum((np.arange(2, 2 * len(self.levels) + 1) - self.sum_entropy())**2 * self.sum)

    def variance(self):

        return np.sum((self.i - np.mean(self.glcm_matrix))**2 * self.glcm_matrix)