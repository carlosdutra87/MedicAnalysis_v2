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