import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glszm_loop import _glszm_loop

image = np.array([[[2, 2, 2, 4, 4, 4, 4],
                   [2, 1, 1, 4, 4, 1, 1],
                   [3, 1, 2, 2, 2, 1, 4],
                   [3, 4, 4, 1, 2, 1, 4],
                   [3, 4, 4, 4, 3, 3, 4],
                   [2, 2, 2, 3, 3, 3, 1],
                   [1, 1, 4, 4, 4, 1, 1]]]).astype(np.uint16)

print _glszm_loop(image)
