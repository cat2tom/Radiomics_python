import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glszm_loop import _glszm_loop

class GLSZM_Matrix:

    def __init__(self, image):

        self.image = image.astype(np.uint16)

        self.glszm_matrix = None
        self.Ng = None
        self.Lz = None

        self.i = None
        self.j = None

        self.ui = None
        self.uj = None

    def calc_GLSZM(self):

        glszm_matrix = _glszm_loop(self.image)

        self.glszm_matrix = glszm_matrix.astype(np.float) / np.sum(glszm_matrix)

        self.Ng = np.arange(1, self.glszm_matrix.shape[0] + 1).astype(np.float)
        self.Lz = np.arange(1, self.glszm_matrix.shape[1] + 1).astype(np.float)

        i, j = np.meshgrid(self.Ng, self.Lz)

        self.i = i.T
        self.j = j.T

        self.ui = np.sum(self.Ng * np.sum(self.glszm_matrix, axis=1))
        self.uj = np.sum(self.Lz * np.sum(self.glszm_matrix, axis=0))

        #print "glszm: \n", self.glszm_matrix
        #print "Ng: ", self.Ng
        #print "Lz: ", self.Lz
        #print "i: ", self.i
        #print "j: ", self.j
        #print "ui: ", self.ui
        #print "uj: ", self.uj

        print "SZE: ", self.small_zone_emphasis()
        print "LZE: ", self.large_zone_emphasis()
        print "GLN: ", self.gray_level_non_uniformity()
        print "ZSN: ", self.zone_size_non_uniformity()
        print "ZP: ", self.zone_percentage()
        print "LGZE: ", self.low_gray_level_zone_emphasis()
        print "HGZE: ", self.high_gray_level_zone_emphasis()
        print "SZLGE: ", self.small_zone_low_gray_level_emphasis()
        print "SZHGE: ", self.small_zone_high_gray_level_emphasis()
        print "LZLGE: ", self.large_zone_low_gray_level_emphasis()
        print "LZHGE: ", self.large_zone_high_gray_level_emphasis()
        print "GLV: ", self.gray_level_variance()
        print "ZSV: ", self.zone_size_variance()

    def small_zone_emphasis(self):

        return np.sum(self.glszm_matrix / (self.j**2))

    def large_zone_emphasis(self):

        return np.sum(self.j**2 * self.glszm_matrix)

    def gray_level_non_uniformity(self):

        return np.sum(np.sum(self.glszm_matrix, axis=1)**2)

    def zone_size_non_uniformity(self):

        return np.sum(np.sum(self.glszm_matrix, axis=0)**2)

    def zone_percentage(self):

        return np.sum(self.glszm_matrix) / np.sum(self.Lz * np.sum(self.glszm_matrix, axis=0))

    def low_gray_level_zone_emphasis(self):

        return np.sum(self.glszm_matrix / self.i**2)

    def high_gray_level_zone_emphasis(self):

        return np.sum(self.i**2 * self.glszm_matrix)

    def small_zone_low_gray_level_emphasis(self):

        return np.sum(self.glszm_matrix / (self.i**2 * self.j**2))

    def small_zone_high_gray_level_emphasis(self):

        return np.sum(self.i**2 * self.glszm_matrix / self.j**2)

    def large_zone_low_gray_level_emphasis(self):

        return np.sum(self.j**2 * self.glszm_matrix / self.i**2)

    def large_zone_high_gray_level_emphasis(self):

        return np.sum(self.i**2 * self.j**2 * self.glszm_matrix)

    def gray_level_variance(self):

        return np.sum((self.i * self.glszm_matrix - self.ui)**2) / (self.Ng[-1] * self.Lz[-1])

    def zone_size_variance(self):

        return np.sum((self.j * self.glszm_matrix - self.uj)**2) / (self.Ng[-1] * self.Lz[-1])


image = np.array([[[2, 2, 2, 4, 4, 4, 4],
                   [2, 1, 1, 4, 4, 1, 1],
                   [3, 1, 2, 2, 2, 1, 4],
                   [3, 4, 4, 1, 2, 1, 4],
                   [3, 4, 4, 4, 3, 3, 4],
                   [2, 2, 2, 3, 3, 3, 1],
                   [1, 1, 4, 4, 4, 1, 1]]])

#print _glszm_loop(image)

glszm_matrix = GLSZM_Matrix(image)
glszm_matrix.calc_GLSZM()

