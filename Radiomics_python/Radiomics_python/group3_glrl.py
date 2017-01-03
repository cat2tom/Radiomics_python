import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glrl_loop import _glrl_vector_loop

from profiling_tools import time

def group3_glrl_features(image):

    SRE = []
    LRE = []
    GLN = []
    RLN = []
    RP = []
    LGLRE = []
    HGLRE = []
    SRLGLE = []
    SRHGLE = []
    LRLGLE = []
    LRHGLE = []
    GLV = []
    RLV = []

    glrl_features = {}

    for i in range(1, 14):

        glrl_matrix = GLRL_Matrix(image, i)

        SRE.append(glrl_matrix.short_run_emphasis())
        LRE.append(glrl_matrix.long_run_emphasis())
        GLN.append(glrl_matrix.gray_level_non_uniformity())
        RLN.append(glrl_matrix.run_length_non_uniformity())
        RP.append(glrl_matrix.run_percentage())
        LGLRE.append(glrl_matrix.low_gray_level_run_emphasis())
        HGLRE.append(glrl_matrix.high_gray_level_rum_emphasis())
        SRLGLE.append(glrl_matrix.short_run_low_gray_level_emphasis())
        SRHGLE.append(glrl_matrix.short_run_high_gray_level_emphasis())
        LRLGLE.append(glrl_matrix.long_run_low_gray_level_emphasis())
        LRHGLE.append(glrl_matrix.long_run_high_gray_level_emphasis())
        GLV.append(glrl_matrix.gray_level_variance())
        RLV.append(glrl_matrix.run_length_variance())

    glrl_features['SRE'] = np.mean(SRE)
    glrl_features['LRE'] = np.mean(LRE)
    glrl_features['GLN'] = np.mean(GLN)
    glrl_features['RLN'] = np.mean(RLN)
    glrl_features['RP'] = np.mean(RP)
    glrl_features['LGLRE'] = np.mean(LGLRE)
    glrl_features['HGLRE'] = np.mean(HGLRE)
    glrl_features['SRLGLE'] = np.mean(SRLGLE)
    glrl_features['SRHGLE'] = np.mean(SRHGLE)
    glrl_features['LRLGLE'] = np.mean(LRLGLE)
    glrl_features['LRHGLE'] = np.mean(LRHGLE)
    glrl_features['GLV'] = np.mean(GLV)
    glrl_features['RLV'] = np.mean(RLV)

    print glrl_features

    return glrl_features

class GLRL_Matrix:

    def __init__(self, image):

        self.image = image.astype(np.uint16)

        self.glrl_matrix = None

        self.Ng = None
        self.Nr = None
        self.Np = None

        self.i = None
        self.j = None

        self.ui = None
        self.uj = None

    def calc_GLRL(self, direction):

        glrl_matrix = _glrl_vector_loop(self.image, direction).astype(np.float)

        self.glrl_matrix = glrl_matrix / np.sum(glrl_matrix)

        self.Ng = np.arange(1, self.glrl_matrix.shape[0] + 1).astype(np.float)
        self.Nr = np.arange(1, self.glrl_matrix.shape[1] + 1).astype(np.float)
        self.Np = float(len(self.image.ravel()))

        i, j = np.meshgrid(self.Ng, self.Nr)

        self.i = i.T
        self.j = j.T

        self.ui = np.sum(self.Ng * np.sum(self.glrl_matrix, axis=1))
        self.uj = np.sum(self.Nr * np.sum(self.glrl_matrix, axis=0))

        print "SRE: ", self.short_run_emphasis()
        print "LRE: ", self.long_run_emphasis()
        print "GLN: ", self.gray_level_non_uniformity()
        print "RLN: ", self.run_length_non_uniformity()
        print "RP: ", self.run_percentage()
        print "LGLRE: ", self.low_gray_level_run_emphasis()
        print "HGLRE: ", self.high_gray_level_rum_emphasis()
        print "SRLGLE: ", self.short_run_low_gray_level_emphasis()
        print "SRHGLE: ", self.short_run_high_gray_level_emphasis()
        print "LRLGLE: ", self.long_run_low_gray_level_emphasis()
        print "LRHGLE: ", self.long_run_high_gray_level_emphasis()
        print "GLV: ", self.gray_level_variance()
        print "RLV: ", self.run_length_variance()

    def short_run_emphasis(self):

        return np.sum(self.glrl_matrix / self.j**2)

    def long_run_emphasis(self):

        return np.sum(self.j**2 * self.glrl_matrix)

    def gray_level_non_uniformity(self):

        return np.sum(np.sum(self.glrl_matrix, axis=1)**2)

    def run_length_non_uniformity(self):

        return np.sum(np.sum(self.glrl_matrix, axis=0)**2)

    def run_percentage(self):

        return np.sum(self.glrl_matrix) / np.sum(self.Nr * np.sum(self.glrl_matrix, axis=0))

    def low_gray_level_run_emphasis(self):

        return np.sum(self.glrl_matrix / self.i**2)

    def high_gray_level_rum_emphasis(self):

        return np.sum(self.i**2 * self.glrl_matrix)

    def short_run_low_gray_level_emphasis(self):

        return np.sum(self.glrl_matrix / (self.i**2 * self.j**2))

    def short_run_high_gray_level_emphasis(self):

        return np.sum((self.i**2 * self.glrl_matrix) / self.j**2)

    def long_run_low_gray_level_emphasis(self):

        return np.sum((self.j**2 * self.glrl_matrix) / self.i**2)

    def long_run_high_gray_level_emphasis(self):

        return np.sum(self.i**2 * self.j**2 * self.glrl_matrix)

    def gray_level_variance(self):

        return np.sum((self.i * self.glrl_matrix - self.ui)**2) / (self.Ng[-1] * self.Nr[-1])

    def run_length_variance(self):

        return np.sum((self.j * self.glrl_matrix - self.uj)**2) / (self.Ng[-1] * self.Nr[-1])
