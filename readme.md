## Hybrid Ant Colony and Particle Swarm Optimization Simulator
### Instructions

1. Install requirements by running the following code in the terminal:
```
pip install -r requirements.txt
```
2. Launch multi_aco_pso using Python 3.7.2 and above interpreter

3. Enter coordinates of vertices using the canvas on the left. Using the buttons on the right, you can change the mode. There are three modes:
    - Add Robot      : Changes the dots you place to represent a robot's start point. Maximum robots are 3 in multi_aco_pso
    - Add End        : Adds an end point. Maximum end point is 1.
    - Add Obstacles  : Adds a vertice of an obstacle. Obstacles are set to 5 vertices for simplicity. Obstacle points must be placed sequentially to form the perimeter of the obstacle.

4.  Once input is entered, press the 'SIMULATE' button.

5. Results can be displayed in the first four buttons of the Simulation Settings panel.
    - Triangulation         : Shows the Delaunay triangular decomposed version of the map. 
    - Dijkstra              : Shows the path of triangular cells chosen as the shortest path, as well as the path connected by using adjacent midpoints from each triangle in the path
    - PSO Optimization      : Shows the optimized path using particle swarm optimization only. 
    - Hybrid Optimization   : Shows the optimized path using the hybrid approach of ant colony and particle swarm optimization.
    - Comparison            : Shows the visual comparison of the paths as well as the comparison of distances.
