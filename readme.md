## Particle Swarm Optimization Simulator
### Instructions

1. Install requirements by running the following code in the terminal:
```
pip install -r requirements.txt
```
2. There are 3 main files in this repository
    - FYP.ipynb      : A notebook for documenting experimentation and development of the particle optimization simulator, may contain outdated functions that are later mofidied in the following 2 files
    - single_pso.py  : A Python script that runs the TKinter GUI. 
    - multi_pso.py   : An extension of the main single PSO that allows for multi robots, however not as accurate as the singular robot version, most likely due to an unresolved bug while implementing multiple robots.

3. Launch single_pso using Python 3.7.2 interpreter

4. Enter coordinates of vertices using the canvas on the right. Using the buttons on the left, you can change the mode. There are three modes:
    - Add Robot      : Changes the dots you place to represent a robot's start point. Maximum robots is 1 in single PSO and 3 in multi PSO
    - Add End        : Adds an end point. Maximum end point for all versions of this application is 1.
    - Add Obstacles  : Adds a vertice of an obstacle. Obstacles are set to 5 vertices for simplicity. Obstacle points must be placed sequentially to form the perimeter of the obstacle.

5.  Once input is entered, press the 'SIMULATE' button.

6. Results can be displayed in the first four buttons of the Simulation Settings panel.
    - Triangulation : Shows the Delaunay triangular decomposed version of the map. 
    - Dijkstra      : Shows the path of triangular cells chosen as the shortest path, as well as the path connected by using adjacent midpoints from each triangle in the path
    - Optimization  : Shows the optimized path using particle swarm optimization. Also has an additional button "SIMULATION" which displays the position of the particles throughout each iteration
    - Comparison    : Shows the visual comparison of both paths as well as the comparison of distances for both paths.
