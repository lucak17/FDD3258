# Assignment 2 Ex 2
import timeit
import array
import numpy as np
from functools import wraps


def timefn(fn):
    timings = []
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timeit.default_timer()
        result = fn(*args, **kwargs)
        t2 = timeit.default_timer()
        elapsed_time = t2 - t1
        timings.append(elapsed_time)
        # print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    
    def get_timings():
        # Returns the list of all recorded timings.
        return timings

    def clear_timings():
        # Clears the list of recorded timings.
        timings.clear()
        
    def get_avg_std():
        m = np.average(timings)
        std = np.std(timings)
        return m,std
        
    measure_time.get_timings = get_timings
    measure_time.clear_timings = clear_timings
    measure_time.get_avg_std = get_avg_std
    
    return measure_time



def dgemm(N,A,B,C):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


def set_lists(N,value_A,value_B,value_C):
    A = [[value_A for _ in range(N)] for _ in range(N)]
    B = [[value_B for _ in range(N)] for _ in range(N)]
    C = [[value_C for _ in range(N)] for _ in range(N)]

    return A,B,C

def set_array(N,value_A,value_B,value_C):
    A = [array.array('d', [value_A] * N) for _ in range(N)] 
    B = [array.array('d', [value_B] * N) for _ in range(N)] 
    C = [array.array('d', [value_C] * N) for _ in range(N)] 

    return A,B,C

def set_np(N,value_A,value_B,value_C):
    A = np.full((N, N), value_A) 
    B = np.full((N, N), value_B)
    C = np.full((N, N), value_C)
    #A = np.random.rand(N, N)
    #B = np.random.rand(N, N)
    #C = np.random.rand(N, N)
            

    return A,B,C



@timefn
def dgemm_lists(N,A,B,C):
    """Matrix multiplication using lists"""
    # C = dgemm(A,B,C)
    return dgemm(N,A,B,C)
    

@timefn
def dgemm_array(N,A,B,C):
    """Matrix multiplication using python array"""
    #C = dgemm(A,B,C)
    return dgemm(N,A,B,C)

@timefn
def dgemm_numpy(N,A,B,C):
    """Matrix multiplication using numpy"""
    #C = dgemm(A,B,C)
    return dgemm(N,A,B,C)
    #return C + np.dot(A, B)

@timefn
def dgemm_matmul(A,B,C):
    """Matrix multiplication using numpy matmul"""
    #C = 
    return C + np.matmul(A,B)



if __name__ == "__main__":

    iterations = 30
    N = 128
    value_A = 1.0
    value_B = 2.0
    value_C = 3.0

    A_np,B_np,C_np = set_np(N,value_A,value_B,value_C)
    for _ in range(iterations):
        A_list,B_list,C_list = set_lists(N,value_A,value_B,value_C)
        A_array,B_array,C_array = set_array(N,value_A,value_B,value_C)
        A_np,B_np,C_np = set_np(N,value_A,value_B,value_C)
        _ = dgemm_lists(N,A_list,B_list,C_list)
        _ = dgemm_array(N,A_array,B_array,C_array)
        _ = dgemm_numpy(N,A_np,B_np,C_np)
        A_np,B_np,C_np = set_np(N,value_A,value_B,value_C)
        _ = dgemm_matmul(A_np,B_np,C_np)
    
    m_lists,std_lists = dgemm_lists.get_avg_std()
    m_array,std_array = dgemm_array.get_avg_std()
    m_np,std_np = dgemm_numpy.get_avg_std()
    m_mat,std_mat = dgemm_matmul.get_avg_std()

    print("Lists avg and std: ", m_lists, " ", std_lists)
    print("Array avg and std: ", m_array, " ", std_array)
    print("Np avg and std: ", m_np, " ", std_np)
    print("Np matmul avg and std: ", m_mat, " ", std_mat)
