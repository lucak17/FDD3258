#cythonMandelbrot.pyx
import numpy as np
cimport numpy as cnp

def mandelbrot(double complex c, int max_iter=100):
    """Computes the number of iterations before divergence."""
    cdef double complex z = 0
    cdef unsigned int n
    for n in range(max_iter):
        if (z.real*z.real + z.imag*z.imag) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter=100):
    """Generates the Mandelbrot set image."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_vals = np.linspace(x_min, x_max, width)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_vals = np.linspace(y_min, y_max, height)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] image = np.empty((height, width),dtype=np.int32)
    cdef int i,j
    cdef double complex c
    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot(c, max_iter)

    return image