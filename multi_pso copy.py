import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
from tkinter import PhotoImage
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import networkx as nx

def initialization():
    global end_point,robots,start_xy,END_XY,OBSTACLES_XY,OBSTACLES,POINTS,co,cf,triangulation,G,shortest_path,ub_x,ub_y,lb_x,lb_y,gbests,midpoints,sp,gb
    end_point = None  # Flag to track if end point is placed
    robots = []
    END_XY = []
    OBSTACLES_XY = []
    OBSTACLES = []
    POINTS = []
    co = []
    cf = []
    triangulation = None
    G = nx.Graph()
    shortest_path = []
    ub_x = []
    lb_x = []
    ub_y = []
    lb_y = []
    gbests = []
    midpoints = []
    sp=[]
    gb=[]
    

initialization()

# Set initial map size
initial_map_size = 900
colors = ["gold","blue","deeppink"]


def draw_obstacle():
    global obstacle_points
    num_points = len(obstacle_points)

    if num_points < 3:
        coord_label.config(text="At least 3 points are needed to form a polygon.")
        return

    # Draw lines connecting the points to form a closed polygon
    for i in range(num_points):
        x1, y1 = obstacle_points[i]
        x2, y2 = obstacle_points[(i + 1) % num_points]
        canvas.create_line(x1, y1, x2, y2, fill="orange")

    # Add the completed polygon to OBSTACLES_XY
    OBSTACLES_XY.append(np.array([[pt[0], initial_map_size - pt[1]] for pt in obstacle_points]))
    obstacle_points = []  # Reset points list

def finalize_obstacle():
    """Finalize the current obstacle when the user clicks a button."""
    draw_obstacle()
    coord_label.config(text="Polygon obstacle finalized. You can start a new obstacle.")    
    return obstacle_points


def draw_path(particle, color='green', label='ACO-PSO'):
    coords = particle.reshape(-1, 2)
    x, y = zip(*coords)
    plt.plot(x, y, marker='o', color=color, label=label)

def get_map_dimensions():
    return initial_map_size, initial_map_size

def get_number_of_robots():
    return len(robots)

def get_obstacle_points():
    return obstacle_points

def reset_button_colors():
    # Reset colors for all buttons
    for btn in [robot_button, end_button, obstacle_button]:
        btn.config(bg='light grey', fg='black', activebackground='grey', activeforeground='white')

# Calculate the midpoints of the edges of a triangle
def calculate_edge_midpoints(triangle):
    edge_midpoints = []
    for i in range(3):
        j = (i + 1) % 3
        midpoint = (triangle[i] + triangle[j]) / 2
        edge_midpoints.append(midpoint)
    return edge_midpoints


def initialize_inputs():
    try:
        global initial_map_size, OBSTACLES_XY, POINTS, OBSTACLES, START_XY, END_XY
        MAP_LENGTH = initial_map_size
        MAP_WIDTH = initial_map_size

        # Define only the 4 boundary points
        MAP = np.array([[0.0, 0.0], [0.0, MAP_WIDTH], [MAP_LENGTH, MAP_WIDTH], [MAP_LENGTH, 0.0]])
        start_xy = robots[0] 
        END_XY = end_point

        OBSTACLES = [None] * len(OBSTACLES_XY)
        POINTS = np.copy(MAP)  # Start with only boundary points

        for i, obs in enumerate(OBSTACLES_XY):
            OBSTACLES[i] = []
            for point in obs:
                # Check if this point already exists in POINTS to avoid duplicates
                exists = np.where(np.all(POINTS == point, axis=1))[0]
                if exists.size > 0:
                    index = exists[0]
                else:
                    POINTS = np.vstack([POINTS, point])
                    index = len(POINTS) - 1
                OBSTACLES[i].append(index)

    except ValueError as e:
        print(f"ValueError: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


        
def cache_obstacle_paths():
    """Convert all obstacles to Path objects once for faster checking."""
    return [Path(obs) for obs in OBSTACLES_XY]
        
def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    px, py = zip(*polygon)

    j = n - 1
    for i in range(n):
        if ((py[i] > y) != (py[j] > y)) and \
           (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-10) + px[i]):
            inside = not inside
        j = i
    return inside

# Function to check if a triangle is contained within any obstacle set
def is_triangle_in_obstacle(triangle_indices, obstacle_paths):
    """Efficient check if triangle centroid is within any obstacle."""
    triangle_coords = POINTS[triangle_indices]
    centroid = np.mean(triangle_coords, axis=0)

    for path in obstacle_paths:
        if path.contains_point(centroid):
            return True
    return False


def triangulate():
    global co, cf, triangulation, POINTS

    co = []  # Triangles inside obstacle
    cf = []  # Triangles in free space

    triangulation = Delaunay(POINTS)
    obstacle_paths = cache_obstacle_paths()

    for triangle in triangulation.simplices:
        if is_triangle_in_obstacle(triangle, obstacle_paths):
            co.append(triangle)
        else:
            cf.append(triangle)

    co[:] = np.array(co)
    cf[:] = np.array(np.unique(cf, axis=0))  # Remove any duplicate triangles

def display_graph(fig,frame):
    graph = FigureCanvasTkAgg(fig,master = frame)
    graph.get_tk_widget().pack(pady=5)
    toolbar = NavigationToolbar2Tk(graph,frame)
    toolbar.update()
    toolbar.pack(anchor='w',fill = tk.X)


def triangulation_window():
    global co, triangulation, colors, sp
    tri = tk.Toplevel()
    tri.title("Triangular Decomposition")
    
    tri_frame = tk.Frame(tri)
    tri_frame.grid(column=1, row=0)
    
    tri_label = tk.Label(tri_frame, text="Triangular Decomposition", font=("Unispace", 16))
    tri_label.pack(pady=10)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all triangles
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    
    # Plot all points
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    # Plot START points (one for each robot)
    for i in robots:
        ax.plot(i[0], i[1], 'o', color=colors[robots.index(i)])
        ax.text(i[0], i[1], ' ROBOT ' + str(robots.index(i) + 1),
                verticalalignment='bottom', horizontalalignment='right',
                color=colors[robots.index(i)], fontweight='bold')

    # Plot END point (common for all)
    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END',
            verticalalignment='top', horizontalalignment='left',
            color='red', fontweight='bold')

    # Highlight triangles inside obstacles
    for triangle in co:
        co_indices = np.array(triangle)
        coords = triangulation.points[co_indices]
        ax.fill(coords[:, 0], coords[:, 1], color='darkorange')  # Solid fill for obstacles

    # --- Plot shortest paths for each robot ---
    # for idx, path in enumerate(sp):
    #     path_points = triangulation.points[np.array(path)]
    #     for i in range(len(path_points) - 1):
    #         start = path_points[i].mean(axis=0)  # Midpoint of triangle vertices
    #         end = path_points[i + 1].mean(axis=0)
    #         ax.plot([start[0], end[0]], [start[1], end[1]],
    #                 color=colors[idx], linestyle='-', linewidth=2,
    #                 label=f'Robot {idx+1} Path' if i == 0 else "")  # Only label first line

    # Handle duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    display_graph(fig, tri_frame)





def dijkstra():
    global shortest_path, G, cf, triangulation, START_XY, END_XY, sp

    G.clear()    # Clear old graph
    sp = []      # Clear old paths

    # Build Graph once
    for simplex in cf:
        triangle_nodes = tuple(sorted(simplex))  # Sort vertices to create unique identifier
        G.add_node(triangle_nodes)

    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2 and len(set(node1).intersection(node2)) == 2:
                triangle1 = triangulation.points[list(node1)]
                triangle2 = triangulation.points[list(node2)]
                distance = np.linalg.norm(triangle1.mean(axis=0) - triangle2.mean(axis=0))
                G.add_edge(node1, node2, weight=distance)

    # Find shortest path for each robot
    for robot_pos in robots:
        start_tri_idx = triangulation.find_simplex(robot_pos)
        goal_tri_idx = triangulation.find_simplex(END_XY)

        if start_tri_idx == -1 or goal_tri_idx == -1:
            sp.append([])  # No valid triangle found
            continue

        start_tri = tuple(sorted(triangulation.simplices[start_tri_idx]))
        goal_tri = tuple(sorted(triangulation.simplices[goal_tri_idx]))

        try:
            path = nx.shortest_path(G, source=start_tri, target=goal_tri, weight='weight')
            sp.append(path)
        except nx.NetworkXNoPath:
            sp.append([])  # No path exists

# Define objective function
def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def obj_function_distance(j,particles):
    global total_distance
    total_distance = distance(robots, (particles[0], particles[1]))
    for i in range(0, len(particles) - 3, 2):
        total_distance += distance((particles[i], particles[i + 1]), (particles[i + 2], particles[i + 3]))
    total_distance += distance(robots[j],[particles[0],particles[1]])
    return total_distance

def dijkstra_window():
    global G, shortest_path, co, midpoints, sp, colors
    dji = tk.Toplevel()
    dji.title("Dijkstra's Algorithm")

    dji_frame = tk.Frame(dji)
    dji_frame.grid(column=1, row=1, padx=10)

    dji_label = tk.Label(dji, text="Dijkstra's Algorithm", font=("Unispace", 16))
    dji_label.grid(column=1, row=0, columnspan=2, pady=10)

    # Initialize midpoints safely
    midpoints.clear()

    # Calculate midpoints for all triangles
    triangle_midpoints = {}
    for triangle in G.nodes:
        triangle_indices = np.array(triangle)
        triangle_points = triangulation.points[triangle_indices]
        edge_midpoints = calculate_edge_midpoints(triangle_points)
        triangle_midpoints[triangle] = edge_midpoints

    for i in range(len(robots)):
        if i >= len(sp) or not sp[i]:
            continue  # Skip if no path for this robot
        
        midpoints.append([])

        # Draw lines through midpoints
        for j in range(len(sp[i]) - 1):
            current_triangle = sp[i][j]
            next_triangle = sp[i][j + 1]
            next_midpoints = None

            for element in triangle_midpoints.get(current_triangle, []):
                if any(np.array_equal(element, next_elem) for next_elem in triangle_midpoints.get(next_triangle, [])):
                    next_midpoints = element
                    break  # Found the common midpoint, break early
            
            if next_midpoints is not None:
                midpoints[i].extend(next_midpoints)

    # Plot the original points and the generated triangles
      # Make figure rectangular
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    # Annotate and plot robots
    for idx, start_pos in enumerate(robots):
        ax.plot(start_pos[0], start_pos[1], 'o', color=colors[idx])
        ax.text(start_pos[0], start_pos[1], f' ROBOT {idx+1}', verticalalignment='bottom', horizontalalignment='right', color=colors[idx], fontweight='bold')

    # Annotate and plot 'END'
    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    # Draw lines connecting midpoints for each robot
    for idx, midpoint in enumerate(midpoints):
        for j in range(0, len(midpoint) - 2, 2):
            ax.plot(
                [midpoint[j], midpoint[j+2]],
                [midpoint[j+1], midpoint[j+3]],
                color=colors[idx], linestyle='--'
            )
        # Connect robot start to first midpoint
        if len(midpoint) >= 2:
            ax.plot(
                [robots[idx][0], midpoint[0]],
                [robots[idx][1], midpoint[1]],
                color=colors[idx], linestyle='--'
            )
            # Connect end to last midpoint
            ax.plot(
                [END_XY[0], midpoint[-2]],
                [END_XY[1], midpoint[-1]],
                color=colors[idx], linestyle='--'
            )

    # Highlight obstacle triangles
    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

    # Display the plot
    display_graph(fig, dji_frame)

    # Create distance data
    data = [
        ["Robot", "Dijkstra's Algorithm\nDistance"],
    ]

    for idx in range(len(robots)):
        if idx < len(midpoints):
            distance = obj_function_distance(idx, midpoints[idx])
            data.append([f"ROBOT {idx+1}", "{:.2f}".format(distance)])

    # Display distance table
    data_frame = tk.Frame(dji)
    data_frame.grid(column=2, row=1, padx=10, pady=10)

    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            label = tk.Label(
                data_frame, text=cell,
                borderwidth=1, relief="solid",
                padx=10, pady=10,
                font=("Helvetica", 16)
            )
            label.grid(row=i, column=j, sticky="nsew")


# Define update methods

def get_next_velocity(v, c1, c2, gb, particles):
    return v + c1 + (c2 * (gb - particles))

def get_next_position(particles, velocity, lb, ub):
    return np.clip(particles + velocity, lb, ub)

def optimize():
    global shortest_path, gb, ub_x, ub_y, lb_x, lb_y, triangulation
    global dijkstra_only_result, pso_result, aco_pso_result, sp, gbests, END_XY

    shortest_path_xy = []
    gbests = [[] for _ in range(len(robots))]
    gb = []
    dijkstra_only_result = []
    pso_result = []
    aco_pso_result = []

    for shortest_path in sp:
            # Change the shortest path array into a numpy array with the coordinates of the triangles instead of the points
            shortest_path_xy.append(triangulation.points[np.array(shortest_path)])
    
    for idx_robot in range(len(sp)):
        # --- Step 1: Setup for each robot ---
        single_path = sp[idx_robot]
        single_path_xy = triangulation.points[np.array(single_path)]
        shortest_path_xy.append(single_path_xy)

        path_edges = []
        for i in range(len(single_path_xy) - 1):
            cur = single_path_xy[i]
            nxt = single_path_xy[i + 1]
            shared = [pt for pt in cur if any(np.array_equal(pt, q) for q in nxt)]
            if len(shared) == 2:
                path_edges.append(shared)

        ub_x = [max(e[0][0], e[1][0]) for e in path_edges]
        lb_x = [min(e[0][0], e[1][0]) for e in path_edges]
        ub_y = [max(e[0][1], e[1][1]) for e in path_edges]
        lb_y = [min(e[0][1], e[1][1]) for e in path_edges]

        lb = np.array([v for pair in zip(lb_x, lb_y) for v in pair])
        ub = np.array([v for pair in zip(ub_x, ub_y) for v in pair])
        d = len(path_edges)
        n = 2 * d

        dijkstra_only_result.append(single_path_xy.flatten().tolist())

        # --- Step 2: PSO-only baseline ---
        def run_pso(seed_particle=None):
            particles = [np.random.uniform(lb, ub) for _ in range(num_particles)]
            if seed_particle is not None:
                particles[0] = seed_particle.copy()
            particles = np.array(particles)
            velocity = np.zeros_like(particles)
            gbest = particles[np.argmin(np.apply_along_axis(lambda e: obj_function_distance(idx_robot, e), 1, particles))]
            for i in range(n_iter):
                idx = np.argmin(np.apply_along_axis(lambda e: obj_function_distance(idx_robot, e), 1, particles))
                cand = particles[idx]
                if obj_function_distance(idx_robot, cand) < obj_function_distance(idx_robot, gbest):
                    gbest = cand.copy()
                c1 = 0.0 * (0.99**i) * np.random.uniform(lb, ub)
                velocity = get_next_velocity(velocity, c1, c2, gbest, particles)
                particles = get_next_position(particles, velocity, lb, ub)
            return gbest

        num_particles = 30
        n_iter = 1000
        c2 = 0.5

        pso_result.append(run_pso().tolist())

        # --- Step 3: ACO part ---
        NUM_ANTS = 50
        ACO_ITER = 100
        rho = 0.2
        offset_lim = 0.5
        alpha, beta = 1.0, 2.0

        straight_lengths = np.array([np.linalg.norm(e[1] - e[0]) for e in path_edges])
        eta = 1.0 / (straight_lengths + 1e-9)
        pheromone = np.ones(len(path_edges))

        best_aco_score = float('inf')
        best_aco_particle = None

        def pick_point_and_edge(i_edge):
            e = path_edges[i_edge]
            ts = np.linspace(0, 1, 6)
            weights = (pheromone[i_edge]**alpha) * (eta[i_edge]**beta)
            weights = np.full(ts.shape, weights)
            weights /= weights.sum()
            t_idx = np.random.choice(len(ts), p=weights)
            t = ts[t_idx]
            base = (1 - t) * e[0] + t * e[1]
            perp = np.array([-(e[1][1]-e[0][1]), e[1][0]-e[0][0]])
            perp /= (np.linalg.norm(perp) + 1e-6)
            offset = np.random.uniform(-offset_lim, offset_lim)
            point = base + offset * perp
            return point, i_edge

        for _ in range(ACO_ITER):
            gen_scores = []
            gen_edge_paths = []

            for ant in range(NUM_ANTS):
                pts = []
                edges_used = []
                for i in range(len(path_edges)):
                    p, used_edge = pick_point_and_edge(i)
                    pts.append(p)
                    edges_used.append(used_edge)
                particle = np.concatenate(pts)
                score = obj_function_distance(idx_robot, particle)
                gen_scores.append(score)
                gen_edge_paths.append(edges_used)

                if score < best_aco_score:
                    best_aco_score = score
                    best_aco_particle = particle.copy()

            pheromone *= (1 - rho)
            best_ant_idx = int(np.argmin(gen_scores))
            deposit_amt = 1.0 / (gen_scores[best_ant_idx] + 1e-9)
            for edge_idx in gen_edge_paths[best_ant_idx]:
                pheromone[edge_idx] += deposit_amt

        # --- Step 4: Final PSO seeded with best ACO particle ---
        aco_pso_result.append(run_pso(seed_particle=best_aco_particle).tolist())



def optimization_window():
    global co, shortest_path, POINTS, triangulation, robots, END_XY, gb, aco_pso_result, sp

    opt = tk.Toplevel()
    opt.title("Hybrid Ant Colony Optimization and Particle Swarm Optimization")

    # Top-level container
    container = tk.Frame(opt)
    container.pack(fill=tk.BOTH, expand=True)

    # Frames side by side (LEFT: plot, RIGHT: info)
    plot_frame = tk.Frame(container)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    info_frame = tk.Frame(container)
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Plot title
    opt_label = tk.Label(plot_frame, text="Hybrid Ant Colony Optimization and Particle Swarm Optimization", font=("Unispace", 16))
    opt_label.pack(pady=10)

    # Make figure rectangular
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    # Plot Start and End points for each robot
    for i in range(len(robots)):
        ax.plot(robots[i][0], robots[i][1], 'o', color='blue')  # Start point for each robot
        ax.text(robots[i][0], robots[i][1], f' ROBOT {i+1}', verticalalignment='bottom', horizontalalignment='right', color='blue', fontweight='bold')

    ax.plot(END_XY[0], END_XY[1], 'o', color='red')  # End point
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    # Path lines for each robot
    for i in range(len(aco_pso_result)):
        result = aco_pso_result[i]
        for j in range(0, len(result) - 2, 2):
            ax.plot([result[j], result[j + 2]], [result[j + 1], result[j + 3]], color='green', linestyle='-', label=f'Robot {i+1} Path' if j == 0 else "")
        ax.plot([robots[i][0], result[0]], [robots[i][1], result[1]], color='green', linestyle='-')
        ax.plot([END_XY[0], result[-2]], [END_XY[1], result[-1]], color='green', linestyle='-')

    # Highlight triangles
    for triangle in shortest_path:
        triangle_indices = np.array(triangle)
        ax.fill(triangulation.points[triangle_indices, 0], triangulation.points[triangle_indices, 1], alpha=0.5, color='lightblue')
    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

    # Display graph in plot_frame
    display_graph(fig, plot_frame)

    # -------------------------------
    # Distance information in info_frame
    # -------------------------------

    # Compute distances for each robot
    robot_distances = [obj_function_distance(i, path) for i, path in enumerate(aco_pso_result)]
    total_distance = sum(robot_distances)

    # Title
    res2_title = tk.Label(info_frame, text="Distances:", font=("Helvetica", 16, "bold"))
    res2_title.pack(pady=(10, 0))

    # Robot distances
    for idx, dist in enumerate(robot_distances):
        robot_label = tk.Label(info_frame, text=f"Robot {idx+1}: {dist:.2f} units", font=("Helvetica", 14))
        robot_label.pack()

    # Total distance
    total_label = tk.Label(info_frame, text=f"Total Distance: {total_distance:.2f} units", font=("Helvetica", 16, "bold"))
    total_label.pack(pady=(10, 10))

    # # -------------------------------
    # # SIMULATION button
    # # -------------------------------
    # simulation_button = tk.Button(info_frame, text="SIMULATION", command=simulation_window,
    #                               font=btn_font, padx=btn_padx, pady=btn_pady,
    #                               relief=btn_relief, bg=btn_bg, fg=btn_fg)
    # simulation_button.pack(pady=10, padx=10, fill='x')





def simulation_window():
    global gbests,co,shortest_path
    plt.close('all')
    sim = tk.Toplevel()
    sim.title("Particle Swarm Optmization Simulation")
    sim_frame = tk.Frame(sim)
    sim_frame.pack()
    gen = tk.Label(sim)
    gen.pack()

    for i in range(len(gbests)):
        gen['text'] = "Generation " + str(i)
        gen.update()
        if i!=0:
            plt.clf()
        x_values = [gbests[i][j] for j in range(0, len(gbests[i]), 2)]
        y_values = [gbests[i][j] for j in range(1, len(gbests[i]), 2)]

        # Plot the original points and the generated triangles
        plt.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
        plt.plot(POINTS[:, 0], POINTS[:, 1], 'o')

        # Annotate and plot 'START' in blue
        plt.plot(START_XY[0], START_XY[1], 'o', color='blue')
        plt.text(START_XY[0], START_XY[1], ' START', verticalalignment='bottom', horizontalalignment='right', color='blue', fontweight='bold')

        # Annotate and plot 'END' in red
        plt.plot(END_XY[0], END_XY[1], 'o', color='red')
        plt.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

        # Highlight the triangles in the shortest path
        for triangle in shortest_path:
            triangle_indices = np.array(triangle)
            plt.fill(triangulation.points[triangle_indices, 0], triangulation.points[triangle_indices, 1], alpha=0.5, color='lightblue')

        for triangle in co:
            co_indices = np.array(triangle)
            plt.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')



        # Plotting the current generation's best solutions
        plt.plot(x_values, y_values,'o')

        plt.pause(0.1)  # Adjust the pause time as needed for animation speed
    plt.draw()
    plt.show()

    display_graph(plt,sim_frame)

def comparison_window():
    global midpoints, POINTS, START_XY, END_XY, shortest_path, co
    global dijkstra_only_result, pso_result, aco_pso_result, robots

    com = tk.Toplevel()
    com.title("Comparison of Path Planning Methods (Multi-Robot)")
    com_frame = tk.Frame(com)
    com_frame.grid(column=2, row=1, padx=10)
    com_label = tk.Label(com, text="Comparison of Pure Dijkstra vs PSO vs ACO+PSO", font=("Unispace", 16))
    com_label.grid(column=1, row=0, columnspan=2, pady=10)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    # Start & End markers
    for start_xy in robots:
        ax.plot(start_xy[0], start_xy[1], 'o', color='blue')
        ax.text(start_xy[0], start_xy[1], ' START', verticalalignment='bottom', horizontalalignment='right', color='blue', fontweight='bold')

    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    # === Draw Paths for each robot ===
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']  # for multiple robots
    line_styles = {'Dijkstra': '--', 'PSO': '-', 'ACO+PSO': '-'}

    for idx_robot in range(len(robots)):
        color = colors[idx_robot % len(colors)]  # cycle through colors

        # 1. Pure Dijkstra path
        dijkstra_path = np.array(dijkstra_only_result[idx_robot])
        for i in range(0, len(dijkstra_path) - 2, 2):
            ax.plot([dijkstra_path[i], dijkstra_path[i + 2]],
                    [dijkstra_path[i + 1], dijkstra_path[i + 3]],
                    color=color, linestyle='--', label=f'Robot {idx_robot+1} Dijkstra' if i == 0 else "")
        ax.plot([robots[idx_robot][0], dijkstra_path[0]], [robots[idx_robot][1], dijkstra_path[1]], color=color, linestyle='--')
        ax.plot([END_XY[0], dijkstra_path[-2]], [END_XY[1], dijkstra_path[-1]], color=color, linestyle='--')

        # 2. Dijkstra + PSO path
        pso_path = np.array(pso_result[idx_robot])
        for i in range(0, len(pso_path) - 2, 2):
            ax.plot([pso_path[i], pso_path[i + 2]],
                    [pso_path[i + 1], pso_path[i + 3]],
                    color=color, linestyle='-', label=f'Robot {idx_robot+1} PSO' if i == 0 else "")
        ax.plot([robots[idx_robot][0], pso_path[0]], [robots[idx_robot][1], pso_path[1]], color=color, linestyle='-')
        ax.plot([END_XY[0], pso_path[-2]], [END_XY[1], pso_path[-1]], color=color, linestyle='-')

        # 3. Dijkstra + ACO + PSO path
        aco_path = np.array(aco_pso_result[idx_robot])
        for i in range(0, len(aco_path) - 2, 2):
            ax.plot([aco_path[i], aco_path[i + 2]],
                    [aco_path[i + 1], aco_path[i + 3]],
                    color=color, linestyle='-', linewidth=2, label=f'Robot {idx_robot+1} ACO+PSO' if i == 0 else "")
        ax.plot([robots[idx_robot][0], aco_path[0]], [robots[idx_robot][1], aco_path[1]], color=color, linestyle='-', linewidth=2)
        ax.plot([END_XY[0], aco_path[-2]], [END_XY[1], aco_path[-1]], color=color, linestyle='-', linewidth=2)

    ax.legend(loc='center right', bbox_to_anchor=(1, 1))
    display_graph(fig, com_frame)

    # === Distance Calculations ===
    res3_frame = tk.Frame(com)
    res3_frame.grid(column=3, row=1, padx=10)

    results_text = ""
    total_reduction_aco = 0

    for idx_robot in range(len(robots)):
        dijkstra_score = obj_function_distance(idx_robot, np.array(dijkstra_only_result[idx_robot]))
        pso_score = obj_function_distance(idx_robot, np.array(pso_result[idx_robot]))
        aco_score = obj_function_distance(idx_robot, np.array(aco_pso_result[idx_robot]))

        diff_pso = dijkstra_score - pso_score
        diff_aco = dijkstra_score - aco_score
        reduction_pso = diff_pso / dijkstra_score * 100
        reduction_aco = diff_aco / dijkstra_score * 100
        total_reduction_aco += reduction_aco

        results_text += (
            f"Robot {idx_robot+1}:\n"
            f"  Pure Dijkstra:\t{dijkstra_score:.2f} units\n"
            f"  Dijkstra + PSO:\t{pso_score:.2f} units\n"
            f"  Dijkstra + ACO+PSO:\t{aco_score:.2f} units\n"
            f"  PSO Decrease:\t\t{diff_pso:.2f} units ({reduction_pso:.2f}%)\n"
            f"  ACO+PSO Decrease:\t{diff_aco:.2f} units ({reduction_aco:.2f}%)\n\n"
        )

    avg_reduction_aco = total_reduction_aco / len(robots)

    res3_label = tk.Label(res3_frame, text=results_text, font=("Helvetica", 14), justify='left')
    res3_label.pack(pady=10, expand=True, fill='both')

    res4_label = tk.Label(res3_frame, text=f"Average ACO+PSO Reduction: {avg_reduction_aco:.2f}%", font=("Unispace", 20), fg="green")
    res4_label.pack(expand=True, fill='both')





def simulate():
    initialize_inputs()
    triangulate()
    dijkstra()
    optimize()
    switch_off(obstacle_button)
    switch_off(finalize_button)
    switch_on(triangulation_button)
    switch_on(dijkstra_button)
    switch_on(optimization_button)
    switch_on(comparison_button)
    coord_label.config(text=" Simulation has been created ! ")

def switch_on(btn):
    if btn['state']=='disabled':
        btn['state']='normal'

def switch_off(btn):
    if btn['state']!='disabled':
        btn['state']='disabled'


def set_robot():
    global current_mode,clear_button
    current_mode = "robot"
    reset_button_colors()
    robot_button.config(bg='lightblue', fg='black')  # Highlight the robot button

def set_end_point():
    global current_mode
    current_mode = "end"
    reset_button_colors()
    end_button.config(bg='lightblue', fg='black')  # Highlight the end button

def set_add_obstacle():
    global current_mode, obstacle_points
    current_mode = "obstacle"
    obstacle_points = []
    reset_button_colors()
    obstacle_button.config(bg='lightblue', fg='black')  # Highlight the obstacle button

# Function to show tooltip on hover
def show_tooltip(event):
    global tooltip
    hovered = False
    for dot in dots:
        x, y, radius, text, dot_id, color = dot
        distance = (event.x - x) ** 2 + (event.y - y) ** 2
        if distance <= (radius + 5) ** 2:
            hovered = True
            if tooltip is None:
                tooltip = tk.Toplevel(root, bg='lightyellow', padx=5, pady=5)
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root}+{event.y_root}")
                tk.Label(tooltip, text=text, bg='lightyellow').pack()
    if not hovered:
        hide_tooltip()

def hide_tooltip():
    global tooltip
    if tooltip:
        tooltip.destroy()
        tooltip = None

def draw_grid():
    global initial_map_size
    for i in range(0, initial_map_size, 20):  # Draw grid lines every 20 units
        canvas.create_line(i, 0, i, initial_map_size, fill="lightgrey", dash=(2, 2))
        canvas.create_line(0, i, initial_map_size, i, fill="lightgrey", dash=(2, 2))

def intro():
    intro = tk.Toplevel()
    intro.iconbitmap("PSO_Logo2.ico")
    intro.title("Beginner's guide")
    intro_frame = tk.Frame(intro,padx=10,pady=10)
    intro_frame.pack()
    intro_label = tk.Label(intro_frame,text= "Particle Swarm Optimization Simulator", font = ("Unispace",16,"bold italic"),padx=10,pady=10)
    intro_label.pack(pady=10)
    message = """   A particle swarm optimization simulator that takes in a set of coordinates for the robots,\n   end points and obstacle points.\n\n    The following constraints are implemented for best performance:\n\n\n
                - Obstacle boundaries are assumed to be extended 
                  beforehand by the size of the robot to ensure seamless
                  movement of robot
                - Obstacles are set as pentagons for simplicity\n
                - Max robots are 3\n
                - Max end point is 1\n"""
    para_label = tk.Label(intro,text=message,font=("Arial Narrow",16),justify='left')
    para_label.pack(pady=10 ,padx=10)


root = tk.Tk()
root.title("ACO-PSO Hybrid Simulator")
# root.iconbitmap("PSO_Logo2.ico")

# intro()

# Create a new frame to hold all sections
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Create two main frames: one for settings, one for simulation
settings_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
settings_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)

simulation_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
simulation_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Label for Simulation Settings frame
settings_label = tk.Label(settings_frame, text="Simulation Settings", font=("Unispace", 16))
settings_label.pack(pady=10)

# Styling
btn_font = tkFont.Font(family="Lexend", size=10)
btn_padx = 10
btn_pady = 10
btn_relief = tk.FLAT
btn_bg = 'light grey'
btn_fg = 'black'



current_mode = "normal"
obstacle_points = []
dots = []  # List to keep track of the dots
tooltip = None  # Tooltip for hover

# Create a label to display the map size
map_size_label = ttk.Label(root, text=f"Map Size: {initial_map_size}x{initial_map_size}")
map_size_label.pack(pady=10)


def update_map_size(value):
    global initial_map_size
    rounded_value = round(float(value))
    map_size_label.config(text=f"Map Size: {rounded_value}x{rounded_value}")
    initial_map_size = rounded_value
    canvas.config(width=initial_map_size, height=initial_map_size)
    draw_grid()  # Redraw the grid when the map size is updated

# Create buttons
button_frame = tk.Frame(settings_frame)
button_frame.pack(side=tk.LEFT, padx=10)

# Adding new disabled buttons at the top of the settings frame

triangulation_button = tk.Button(button_frame, text="TRIANGULATION", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg, command=triangulation_window)
# triangulation_button.pack(pady=5,padx=5,fill='x')

dijkstra_button = tk.Button(button_frame, text="DIJKSTRA", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command = dijkstra_window)
# dijkstra_button.pack(pady=5,padx=5,fill='x')

optimization_button = tk.Button(button_frame, text="OPTIMIZATION", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command= optimization_window)
# optimization_button.pack(pady=5,padx=5,fill='x')

comparison_button = tk.Button(button_frame, text="COMPARISON", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command=comparison_window)
# comparison_button.pack(pady=5,padx=5,fill='x')

robot_button = tk.Button(button_frame, text="ADD ROBOT", command=set_robot, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
# robot_button.pack(pady=5,padx=5,fill='x')

end_button = tk.Button(button_frame, text="ADD END POINT", command=set_end_point, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
# end_button.pack(pady=5,padx=5,fill='x')

obstacle_button = tk.Button(button_frame, text="ADD OBSTACLE", command=set_add_obstacle, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
# obstacle_button.pack(pady=5,padx=5,fill='x')

finalize_button = tk.Button(button_frame, text="FINALIZE OBSTACLE", command=finalize_obstacle, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
# finalize_button.pack(pady=5,padx=5,fill='x')

robot_button.grid(row=1, column=0, padx=5, pady=5)
end_button.grid(row=1, column=1, padx=5, pady=5)
obstacle_button.grid(row=1, column=2, padx=5, pady=5)
finalize_button.grid(row=1, column=3, padx=5, pady=5)

triangulation_button.grid(row=1, column=5, padx=5, pady=5)
dijkstra_button.grid(row=1, column=6, padx=5, pady=5)
optimization_button.grid(row=1, column=7, padx=5, pady=5)
comparison_button.grid(row=1, column=8, padx=5, pady=5)


# Label to display coordinates
coord_label = tk.Label(main_frame, text="", font=tkFont.Font(family="Helvetica", size=10))
coord_label.grid(row=0, column=1, padx=1, pady=5)



def clear_canvas():
    global obstacle_points,current_mode,end_point,robots,clear_button
    robots = []
    current_mode = "normal"
    canvas.delete("all")
    obstacle_points = []
    reset_button_colors()
    end_point = None  # Flag to track if end point is placed
    coord_label.config(text="Coordinates have been cleared.")
    draw_grid()
    switch_off(simulate_button)
    switch_off(clear_button)
    switch_on(robot_button)
    switch_on(end_button)
    switch_on(obstacle_button)
    switch_on(finalize_button)
    switch_off(triangulation_button)
    switch_off(dijkstra_button)
    switch_off(optimization_button)
    switch_off(comparison_button)
    initialization()

clear_button = tk.Button(button_frame, text="CLEAR", command=clear_canvas,state='disabled', font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg='grey', fg='white')
clear_button.grid(row=1, column=9, padx=5, pady=5)

# Add the SIMULATE button
simulate_button = tk.Button(button_frame, text="SIMULATE", command=simulate,state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
# simulate_button.pack(pady=10,padx=10,fill='x')
simulate_button.grid(row=1, column=4, padx=5, pady=5)

def on_canvas_click(event):
    global dots,robots,end_point,initial_map_size
    x, y = event.x, event.y
    radius = 3  # Radius of the dot
    coord_text = f"({x}, {initial_map_size-y})"

    if current_mode == "normal":
        coord_label.config(text="Please select a mode")
        return  # Exit the function if mode is "normal"
    
    # Set the color based on the current mode
    color = "black"
    if current_mode == "robot":
        if (len(robots)==3):
            coord_label.config(text="A maximum of 3 robot can be placed")
            return
        else:
            color = "blue"
            robots.append([x,initial_map_size-y])
            coord_label.config(text=str(current_mode).capitalize()+" "+str(len(robots))+" has been placed at "+coord_text+" !")
            # canvas.create_image(x, y, image=robot_icon, anchor="center")
            # switch_off(robot_button)
    elif current_mode == "end":
        if end_point==None:
            color = "red"
            end_point = (x,initial_map_size-y)
            # Update coordinate label
            coord_label.config(text=str(current_mode).capitalize()+" point has been placed at "+coord_text+" !")
        else:
            coord_label.config(text="Only one end point can be placed")
            return  # Do not add another end point
    elif current_mode == "obstacle":
        color = "orange"
        obstacle_points.append([x, y])
    # Update coordinate label
        coord_label.config(text=str(current_mode).capitalize()+" point has been placed at "+coord_text+" !")
      
    dot_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
    dots.append((x, y, radius, coord_text, dot_id, color))

    switch_on(clear_button)

    if (len(robots)==3):
        switch_off(robot_button)

    if (end_point!=None):
        switch_off(end_button)

    if (len(robots)<=3 and end_point!=None):
            switch_on(simulate_button)


# Label for Simulation frame
simulation_label = tk.Label(simulation_frame, text="Map", font=("Unispace", 16))
simulation_label.pack(pady=10)

# Set canvas size to 1000x1000 and create event bindings
canvas = tk.Canvas(simulation_frame, width=initial_map_size, height=initial_map_size, bg='grey')
canvas.pack()
canvas.bind("<Button-1>", on_canvas_click)
canvas.bind("<Motion>", show_tooltip)
canvas.bind("<Leave>", lambda e: hide_tooltip())

#Load Icons
robot_icon = PhotoImage(file="robot.png") 
# flag_icon = PhotoImage(file="flag_icon.png")    



# Draw grid lines
draw_grid()

# Create a slider
# slider_label = ttk.Label(settings_frame, text="Adjust Map Size:")
# slider_label.pack()
# slider = ttk.Scale(settings_frame, from_=10, to=1000, orient="horizontal", command=update_map_size)
# slider.set(initial_map_size)  # Set initial value
# slider.pack(pady=10)

root.mainloop()
