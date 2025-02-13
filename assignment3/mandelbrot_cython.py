import numpy as np
import matplotlib.pyplot as plt
import cythonMandelbrot
import timeit

# Parameters
width, height = 1000, 800
x_min, x_max, y_min, y_max = -2, 1, -1, 1

# Generate fractal
t1 = timeit.default_timer()
image = cythonMandelbrot.mandelbrot_set(width, height, x_min, x_max, y_min, y_max)
t2 = timeit.default_timer()

print("Time to soluton :", t2-t1)

# Display
plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()
