import numpy as np
import cythonFn
import timeit


def initialize_x(gridX,gridY):
    
    x = np.random.rand(gridX,gridY)
    x[:,0] =  0
    x[:,-1] = 0
    x[0,:] = 0
    x[-1,:] = 0 

    return x

"""
def gauss_seidel(f):
    newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] + newf[i+1,j] + newf[i-1,j])
    
    return newf
"""

if __name__ == "__main__":
    
    iterations = 1000
    gridX = 512
    gridY = gridX

    x = initialize_x(gridX,gridY)

    t1 = timeit.default_timer()
    for i in range(iterations):    
        x = cythonFn.gauss_seidel(x)

    t2 = timeit.default_timer()

    print("Time to soluton :", t2-t1)