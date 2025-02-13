#cythonFn.pyx
import numpy as np
cimport numpy as cnp

def gauss_seidel(f):
    newf = f.copy()
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] + newf[i+1,j] + newf[i-1,j])
    
    return newf


def gauss_seidel_np(cnp.ndarray[cnp.float64_t, ndim=2] f):
    cdef cnp.ndarray[cnp.float64_t, ndim=2]  newf = f.copy()
    cdef unsigned int i,j
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] + newf[i+1,j] + newf[i-1,j])
    
    return newf

