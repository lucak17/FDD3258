import numpy as np
import matplotlib.pyplot as plt
import timeit
import dask
from dask.distributed import Client
import dask.array as da
import os
import vtk

def write_flowMap_to_vtk_structured(u_velocity, v_velocity, step):
    """
    Write the 2D flowMap as a structured grid to a VTK file.
    """
    output_dir="vtk_outputs"
    os.makedirs(output_dir, exist_ok=True)
    nx, ny = u_velocity.shape

    points = vtk.vtkPoints()

    sg = vtk.vtkStructuredGrid()
    sg.SetDimensions(nx, ny, 1)

    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(float(i), float(j), 0.0)
    sg.SetPoints(points)

    field = vtk.vtkDoubleArray()
    field.SetName("flowMap")
    field.SetNumberOfComponents(3) 

    for j in range(ny):
        for i in range(nx):
            field.InsertNextTuple3((u_velocity[i, j]), (v_velocity[i, j]), 0.0)
    sg.GetPointData().SetVectors(field)

    writer = vtk.vtkXMLStructuredGridWriter()
    filename = os.path.join(output_dir, f"flowMap_{step:03d}.vts")
    writer.SetFileName(filename)
    writer.SetInputData(sg)
    writer.Write()

def laplacian(field):
    """Computes the discrete Laplacian of a 2D field using finite differences."""
    return( np.roll(field, shift=1, axis=0) + np.roll(field, shift=-1, axis=0) +
           np.roll(field, shift=1, axis=1) + np.roll(field, shift=-1, axis=1) - 
           4 * field )

def laplacianDask(field):
    """Creates the dask map_overlap with laplacian stencil."""
    return da.map_overlap(laplacian, field, depth=1, boundary="periodic")

def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
    """Updates ocean velocity and temperature fields using a simplified flow model."""    
    u_new = u + alpha * laplacianDask(u) + beta * wind
    v_new = v + alpha * laplacianDask(v) + beta * wind
    temperature_new = temperature + 0.01 * laplacianDask(temperature)  # Small diffusion
    return u_new, v_new, temperature_new


if __name__ == "__main__":

    # Grid size 
    GRID_SIZE = 200
    DA_CHUNK = 100
    TIME_STEPS = 100

    client = Client(processes=True)
    print("Dask Dashboard running at:", client)
    
    # Initialize temperature field (random values between 5C and 30C)
    temperature_dask = da.random.uniform(5, 30, (GRID_SIZE, GRID_SIZE), chunks=(DA_CHUNK, DA_CHUNK))

    # Initialize velocity fields (u: x-direction, v: y-direction)
    u_velocity_dask = da.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE), chunks=(DA_CHUNK, DA_CHUNK))
    v_velocity_dask = da.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE), chunks=(DA_CHUNK, DA_CHUNK))

    # Initialize wind influence (adds turbulence)
    wind_dask = da.random.uniform(-0.5, 0.5, (GRID_SIZE, GRID_SIZE), chunks=(DA_CHUNK, DA_CHUNK))

    # Run the simulation
    t1 = timeit.default_timer()
    for t in range(TIME_STEPS):
        u_velocity_dask, v_velocity_dask, temperature_dask = update_ocean(u_velocity_dask, v_velocity_dask, temperature_dask, wind_dask)
        if t % 10 == 0 or t == TIME_STEPS - 1:
            u_velocity_dask, v_velocity_dask, temperature_dask = dask.persist(u_velocity_dask, v_velocity_dask, temperature_dask)
            # write VTK
            u_velocity, v_velocity, temperature = dask.compute(u_velocity_dask, v_velocity_dask, temperature_dask)
            write_flowMap_to_vtk_structured(u_velocity, v_velocity, t)
            print(f"Time Step {t}: Ocean currents updated.")
    
    u_velocity, v_velocity, temperature = dask.compute(u_velocity_dask, v_velocity_dask, temperature_dask)

    t2 = timeit.default_timer()
    print("Time to soluton :", t2-t1)

    # write VTK
    #write_flowMap_to_vtk_structured(u_velocity, v_velocity, TIME_STEPS )
    
    # Plot the velocity field
    plt.figure(figsize=(6, 5))
    plt.quiver(u_velocity[::10, ::10], v_velocity[::10, ::10])
    plt.title("Ocean Current Directions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

    # Plot temperature distribution
    plt.figure(figsize=(6, 5))
    plt.imshow(temperature, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Temperature (°C)")
    plt.title("Ocean Temperature Distribution")
    plt.show()
    client.close()
    print("Simulation complete.")