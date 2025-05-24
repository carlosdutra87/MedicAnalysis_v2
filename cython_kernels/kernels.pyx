# cython_kernels/kernels.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np           
cimport numpy as cnp            
from libc.stdint cimport uint8_t, uint16_t
from cython cimport double

cpdef double fast_mean_pixels(
        cnp.ndarray[cnp.uint16_t, ndim=2] img,
        cnp.ndarray[cnp.int32_t, ndim=1] xs,
        cnp.ndarray[cnp.int32_t, ndim=1] ys):
    """
    Compute mean over points (xs[i], ys[i]) in a single 2D uint16 image.
    """
    cdef Py_ssize_t n = xs.shape[0]
    cdef Py_ssize_t i
    cdef double acc = 0.0

    for i in range(n):
        acc += img[ ys[i], xs[i] ]
    return acc / n

cpdef cnp.ndarray[cnp.uint8_t, ndim=3] fast_temporal_average(
        cnp.ndarray[cnp.uint8_t, ndim=3] stack,
        int window):
    """
    Apply a moving‐average of length `window` over stack[T,H,W] uint8,
    returning a new uint8 array of shape (T-window+1, H, W).
    """
    cdef int T = stack.shape[0]
    cdef int H = stack.shape[1]
    cdef int W = stack.shape[2]
    cdef int t, h, w, k
    cdef int sum_pix

    cdef cnp.ndarray out_arr = np.empty((T - window + 1, H, W), dtype=np.uint8)

    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out_mv = out_arr

    for t in range(0, T - window + 1):
        for h in range(H):
            for w in range(W):
                sum_pix = 0
                for k in range(window):
                    sum_pix += stack[t + k, h, w]
                out_mv[t, h, w] = <uint8_t>(sum_pix // window)

    return out_mv

cpdef cnp.ndarray[cnp.uint8_t, ndim=3] normalize_stack(
        cnp.ndarray[cnp.uint16_t, ndim=3] stack,
        double p_low,
        double p_high):
    """
    Normalize a stack of uint16 images to uint8 using percentile thresholds.
    """
    import numpy as np 
    cdef int T = stack.shape[0]
    cdef int H = stack.shape[1]
    cdef int W = stack.shape[2]
    cdef int t, h, w
    cdef double min_val, max_val, val, norm_val
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out = np.empty((T, H, W), dtype=np.uint8)

    flat = stack[0].ravel()
    min_val = np.percentile(flat, p_low)
    max_val = np.percentile(flat, p_high)

    for t in range(T):
        for h in range(H):
            for w in range(W):
                val = stack[t, h, w]
                if val < min_val:
                    norm_val = 0
                elif val > max_val:
                    norm_val = 255
                else:
                    norm_val = 255.0 * (val - min_val) / (max_val - min_val)
                out[t, h, w] = <uint8_t>norm_val

    return out

cpdef cnp.ndarray[cnp.uint8_t, ndim=2] fast_apply_ctbv(
        cnp.ndarray[cnp.uint8_t, ndim=2] img,
        double contrast, double brightness,
        int thr_min, int thr_max):
    """
    Apply contrast, brightness, and threshold to one 2D frame in-place.
    """
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int h, w
    cdef double v
    for h in range(H):
        for w in range(W):
            v = img[h, w] * (1.0 + contrast) + brightness
            if v < thr_min:
                img[h, w] = 0
            elif v > thr_max:
                img[h, w] = 255
            else:
                img[h, w] = <uint8_t>v
    return img

cpdef cnp.ndarray[cnp.uint8_t, ndim=2] fast_invert_image(
        cnp.ndarray[cnp.uint8_t, ndim=2] img):
    """
    Invert a 2D uint8 image (255 - pixel_value).
    """
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int h, w
    cdef uint8_t val
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] out = np.empty_like(img, dtype=np.uint8)

    for h in range(H):
        for w in range(W):
            val = img[h, w]
            out[h, w] = 255 - val 
    return out

cdef bint _is_point_in_polygon(double x, double y,
                               cnp.ndarray[cnp.double_t, ndim=1] poly_x,
                               cnp.ndarray[cnp.double_t, ndim=1] poly_y):
    """
    Internal helper: Check if a single point (x, y) is inside the polygon.
    Based on the ray casting algorithm.
    """
    cdef Py_ssize_t nb_corners = poly_x.shape[0]
    cdef Py_ssize_t i, j
    cdef bint odd_nodes = False
    cdef double xi, yi, xj, yj

    j = nb_corners - 1
    for i in range(nb_corners):
        xi = poly_x[i]
        yi = poly_y[i]
        xj = poly_x[j]
        yj = poly_y[j]

        if (yi < y and yj >= y) or (yj < y and yi >= y):
            if (xi + (y - yi) / (yj - yi) * (xj - xi) < x):
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

cpdef tuple get_points_in_polygon(
    cnp.ndarray[cnp.double_t, ndim=1] segment_x_points,
    cnp.ndarray[cnp.double_t, ndim=1] segment_y_points,
    int img_width, int img_height): 
    """
    Get the list of points (x, y) inside the polygon defined by segment_x_points, segment_y_points.
    Returns two NumPy arrays: (xs_in, ys_in).
    """
    cdef int left = np.floor(np.min(segment_x_points))
    cdef int right = np.ceil(np.max(segment_x_points))
    cdef int top = np.ceil(np.max(segment_y_points)) # max y é o limite inferior 
    cdef int bot = np.floor(np.min(segment_y_points)) # min y é o limite superior

    left = max(0, left)
    right = min(img_width - 1, right)
    bot = max(0, bot)
    top = min(img_height - 1, top)

    cdef list res_x_list = []
    cdef list res_y_list = []
    cdef int x, y

    for x in range(left, right + 1): # Inclui o limite superior
        for y in range(bot, top + 1): # Inclui o limite superior
            if _is_point_in_polygon(<double>x, <double>y, segment_x_points, segment_y_points):
                res_x_list.append(x)
                res_y_list.append(y)

    return (np.array(res_x_list, dtype=np.int32), np.array(res_y_list, dtype=np.int32))