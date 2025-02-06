import pytest
from JuliaSet0 import calc_pure_python


def get_test_data():
    return [(1000, 300, 33219980), (100, 300, 12564342), (100, 100, 332422)]

@pytest.mark.parametrize('width, max_iterations, expected_sum',get_test_data())
def test_julia_set_general(width, max_iterations,expected_sum):
    """Test the Julia set computation with variable grid size and iterations."""
    output = calc_pure_python(width, max_iterations)    
    assert sum(output) == expected_sum