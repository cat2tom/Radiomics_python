import numpy as np
cimport numpy as np
cimport cython

def _boundary_check(int xs, int ys, int zs, int len_x, int len_y, int len_z):

    if xs >= 0 and xs < len_x and ys >= 0 and ys < len_y and zs >= 0 and zs < len_z:

        return True

    else:

        return False

def _calc_size_zone(int i, int x0, int y0, int z0,
                          unsigned short [:,:,:] image, 
                          unsigned short [:,:,:] masked_array,
                          unsigned short [:,:] out):

    size = len(_explore_neighborhood(i, x0, y0, z0, image, masked_array))

    out[i-1, size-1] += 1

@cython.boundscheck(False)
def _explore_neighborhood(int i, int x0, int y0, int z0,
                          unsigned short [:,:,:] image, 
                          unsigned short [:,:,:] masked_array):

    cdef:
        int j
        int xs, ys, zs, len_x, len_y, len_z
        int dx, dy, dz

    len_z = image.shape[0]
    len_y = image.shape[1]
    len_x = image.shape[2]

    neighbors = [[x0, y0, z0]]

    masked_array[z0, y0, x0] = 1

    for dx in range(-1, 2):

        for dy in range(-1, 2):

            for dz in range(-1, 2):

                if dx == 0 and dy == 0 and dz == 0: 
                
                    continue
                
                xs = x0 + dx
                ys = y0 + dy
                zs = z0 + dz

                if not _boundary_check(xs, ys, zs, len_x, len_y, len_z):

                    continue

                j = image[zs, ys, xs]

                if not j == i:

                    continue

                if masked_array[zs, ys, xs] == 0:

                    masked_array[zs, ys, xs] == 1

                    neighbors.extend(_explore_neighborhood(i, xs, ys, zs, image, masked_array))

    return neighbors

def _form_np_matrix(np.ndarray out):

    cdef int i, val, idx

    sum_col = np.sum(out, axis=0)

    for i, val in enumerate(sum_col[::-1]):

        if val != 0:

            break

    return out[:,0:-i]

@cython.boundscheck(False)
def _glszm_loop(unsigned short [:,:,:] image):

    cdef:
        int i, x, y, z, len_x, len_y, len_z
        int levels, max_len_of_out

    len_z = image.shape[0]
    len_y = image.shape[1]
    len_x = image.shape[2]

    levels = np.max(image)
    max_len_of_out = len_x * len_y * len_z

    out = np.zeros([levels, max_len_of_out], dtype=np.uint16)
    cdef unsigned short [:,:] out_view = out

    masked_array = np.zeros([len_z, len_y, len_x], dtype=np.uint16)
    cdef unsigned short [:,:,:] masked_view = masked_array

    for z in range(len_z):

        for y in range(len_y):

            for x in range(len_x):

                i = image[z, y, x]

                if i == 0:

                    masked_view[z, y, x] == 1

                    continue

                if masked_view[z, y, x] == 0:

                    _calc_size_zone(i, x, y, z, image, masked_view, out_view)

    return _form_np_matrix(out)