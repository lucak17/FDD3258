import pytest
from JuliaSetA2E1 import calc_pure_python

@pytest.mark.parametrize("grid_size, iterations, expected_sum", [
    (1000, 300, 33219980),  # Default test case
    (500, 100, 3278808),    # Example alternate test case (update expected sum)
])
def test_julia_set(grid_size, iterations, expected_sum):
    result = calc_pure_python(desired_width=grid_size, max_iterations=iterations)
    assert result == expected_sum, f"Failed for grid={grid_size}, iterations={iterations}"
