import numpy as np
import matplotlib.pyplot as plt
import random
import timeit 
import os
import vtk


# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 


def write_forest_to_vtk_structured(forest, day):
    """
    Write the 2D forest grid (NumPy array) as a structured grid to a VTK file.
    """
    output_dir="vtk_outputs"
    os.makedirs(output_dir, exist_ok=True)
    nx, ny = forest.shape

    # Create a vtkStructuredGrid object
    sg = vtk.vtkStructuredGrid()
    sg.SetDimensions(nx, ny, 1)

    # Create vtkPoints
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(float(i), float(j), 0.0)
    sg.SetPoints(points)

    # Create an array to hold the scalar data
    scalars = vtk.vtkIntArray()
    scalars.SetName("wildFire")
    scalars.SetNumberOfComponents(1)
    
    # Insert the forest state values
    for j in range(ny):
        for i in range(nx):
            scalars.InsertNextValue(int(forest[i, j]))
    sg.GetPointData().SetScalars(scalars)

    # Write the structured grid to a VTK file
    writer = vtk.vtkXMLStructuredGridWriter()
    filename = os.path.join(output_dir, f"forest_day_{day:03d}.vts")
    writer.SetFileName(filename)
    writer.SetInputData(sg)
    writer.Write()
    
    
def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def simulate_wildfire():
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
        #write_forest_to_vtk_structured(forest, day)
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
        
        """
        # Plot grid every 5 days
        if day == DAYS - 1:
        #if day % 5 == 0 or day == DAYS - 1:
            plt.figure(figsize=(6, 6))
            plt.imshow(forest, cmap='viridis', origin='upper')
            plt.title(f"Wildfire Spread - Day {day}")
            plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
            plt.show()
        """
        
    return fire_spread


if __name__ == "__main__":

    numSimulations = 16
    # Run simulation
    t1 = timeit.default_timer()
    results = np.zeros((numSimulations,DAYS))
    for i in range(numSimulations):
        #fire_spread_over_time = simulate_wildfire()
        resultLocal = simulate_wildfire()
        results[i,:len(resultLocal)] = resultLocal
        

    resultsAvg = np.mean(results,0)

    t2 = timeit.default_timer()

    print("Time to soluton :", t2-t1)


    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(resultsAvg)), resultsAvg, label="Burning Trees")
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title(f"Average wildfire Spread Over Time over {numSimulations} runs")
    plt.legend()
    plt.show()