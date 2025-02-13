import numpy as np
import matplotlib.pyplot as plt
import timeit 

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    x_2d = np.tile(x_vals, (height,1))
    y_2d = np.tile(y_vals.reshape(-1,1), (1,width))
    print(x_2d.shape)
    print(y_2d.shape)

    image = np.zeros((height, width))
    mask = np.full((height,width), True, dtype = bool)

    x_2d0 = x_2d.copy()
    y_2d0 = y_2d.copy()

    for i in range(max_iter):
      if not mask.any():
        break
      tmp = x_2d.copy()
      x_2d[mask] = np.multiply(x_2d[mask],x_2d[mask]) - np.multiply(y_2d[mask],y_2d[mask]) + x_2d0[mask]
      y_2d[mask] = 2*np.multiply(tmp[mask],y_2d[mask]) + y_2d0[mask]
      image[mask] +=1
      mask = ( ( np.multiply(x_2d,x_2d) + np.multiply(y_2d,y_2d) )< 4 )
    return image
# Parameters
width, height = 1000, 800
x_min, x_max, y_min, y_max = -2, 1, -1, 1

# Generate fractal
t1 = timeit.default_timer()
image = mandelbrot_set(width, height, x_min, x_max, y_min, y_max)
t2 = timeit.default_timer()

print("Time to soluton :", t2-t1)

# Display
plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()