import pytest
from JuliaSet0 import calc_pure_python


def test_julia_set_fixed():
    """Test the Julia set computation with fixed grid size and iterations."""
    width = 1000  # Grid size
    max_iterations = 300  # Number of iterations

    output = calc_pure_python(width, max_iterations)
    
    # Expected sum for a 1000x1000 grid with 300 iterations
    expected_sum = 33219980
    
    assert sum(output) == expected_sum