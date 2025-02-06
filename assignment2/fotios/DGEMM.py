import numpy as np
import time
import pytest

def dgemm_numpy(A, B, C):
    """DGEMM implementation using NumPy."""
    return C + np.dot(A, B)

def measure_execution_time(func, A, B, C):
    """Measure execution time of matrix multiplication."""
    start_time = time.perf_counter()
    D = func(A, B, C)
    end_time = time.perf_counter()
    return D, end_time - start_time

def calculate_flops(N, execution_time):
    """Calculate FLOPS (floating point operations per second)."""
    operations = 2 * (N ** 3)  # DGEMM performs 2*N^3 operations
    return operations / execution_time

def run_benchmark(runs=10):
    """Run the DGEMM benchmark for different matrix sizes and compare performance."""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    results = []
    
    for N in sizes:
        times_numpy = []
        times_matmul = []
        
        for _ in range(runs):
            A = np.random.rand(N, N)
            B = np.random.rand(N, N)
            C = np.random.rand(N, N)
            
            # Measure time for DGEMM with NumPy
            _, time_numpy = measure_execution_time(dgemm_numpy, A, B, C)
            times_numpy.append(time_numpy)
            
            # Measure time for NumPy matmul
            _, time_matmul = measure_execution_time(np.matmul, A, B, C)
            times_matmul.append(time_matmul)
        
        avg_time_numpy = np.mean(times_numpy)
        std_time_numpy = np.std(times_numpy)
        flops_numpy = calculate_flops(N, avg_time_numpy)
        
        avg_time_matmul = np.mean(times_matmul)
        std_time_matmul = np.std(times_matmul)
        flops_matmul = calculate_flops(N, avg_time_matmul)
        
        results.append((N, avg_time_numpy, std_time_numpy, flops_numpy, avg_time_matmul, std_time_matmul, flops_matmul))
    
    return results

def test_dgemm_correctness():
    """Unit test to verify correctness of DGEMM implementation."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = np.array([[9.0, 10.0], [11.0, 12.0]])
    expected = C + np.matmul(A, B)
    result = dgemm_numpy(A, B, C)
    np.testing.assert_allclose(result, expected, atol=1e-6)

if __name__ == "__main__":
    benchmark_results = run_benchmark()
    
    print("Matrix Size | Avg Time DGEMM (s) | Std Dev DGEMM | FLOPS DGEMM | Avg Time MatMul (s) | Std Dev MatMul | FLOPS MatMul")
    print("-" * 100)
    for res in benchmark_results:
        print(f"{res[0]:<11} | {res[1]:<17.6f} | {res[2]:<14.6f} | {res[3]:<12.2e} | {res[4]:<17.6f} | {res[5]:<14.6f} | {res[6]:<12.2e}")
    
    # Run pytest to validate correctness
    pytest.main(["-v", "--tb=short", "-q", "DGEMM.py"])

