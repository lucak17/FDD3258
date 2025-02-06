import pytest
from dgemm import  set_lists,dgemm_lists, set_array,dgemm_array, set_np, dgemm_numpy, dgemm



@pytest.mark.parametrize("N", [100])
def test_dgemm_lists(N):
    """Test dgemm_lists"""
    value_A = 1.0
    value_B = 2.0
    value_C = 3.0 
    A,B,C = set_lists(N,value_A,value_B,value_C)
    output = dgemm_lists(N,A,B,C)
    for i in range(N):
        for j in range(N):
            assert(output[i][j] == 3.0 + N*2) 

    
@pytest.mark.parametrize("N", [100])
def test_dgemm_array(N):
    """Test dgemm_array"""
    value_A = 1.0
    value_B = 2.0
    value_C = 3.0 
    A,B,C = set_array(N,value_A,value_B,value_C)
    output = dgemm_array(N,A,B,C)
    for i in range(N):
        for j in range(N):
            assert(output[i][j] == 3.0 + N*2)

    
@pytest.mark.parametrize("N", [100])
def test_dgemm_numpy(N):
    """Test dgemm_numpy"""
    value_A = 1.0
    value_B = 2.0
    value_C = 3.0 
    A,B,C = set_np(N,value_A,value_B,value_C)
    output = dgemm_numpy(N,A,B,C)
    for i in range(N):
        for j in range(N):
            assert(output[i][j] == 3.0 + N*2)
    