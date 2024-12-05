import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import networkx as nx


def initialization():
    global end_point,robots,start_xy,END_XY,OBSTACLES_XY,OBSTACLES,POINTS,co,cf,triangulation,G,shortest_path,ub_x,ub_y,lb_x,lb_y,gbests,midpoints,sp,gb
    end_point = None  # Flag to track if end point is placed
    robots = []
    start_xy = []
    END_XY = []
    OBSTACLES_XY = []
    OBSTACLES = []
    POINTS = []
    co = []
    cf = []
    triangulation = None
    G = nx.Graph()
    shortest_path = []
    gbests = []
    midpoints = []
    sp = []
    gb = []
    

initialization()


# Set initial map size
initial_map_size = 900
colors = ["gold","blue","deeppink"]


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
        global initial_map_size,OBSTACLES_XY,POINTS,OBSTACLES,start_xy,END_XY
        MAP_LENGTH = initial_map_size
        MAP_WIDTH = initial_map_size

        # Define the coordinates of the points for the map boundaries
        MAP = np.array([[0.0, 0.0], [0.0, MAP_WIDTH], [MAP_LENGTH, MAP_WIDTH], [MAP_LENGTH, 0.0]])
        start_xy = robots[0] 
        END_XY = end_point

        # Define list of obstacle indices
        OBSTACLES = [None] * len(OBSTACLES_XY)

        hull = ConvexHull(MAP)
        POINTS = MAP[hull.vertices]

        for obs in OBSTACLES_XY:
            POINTS = np.append(POINTS, obs, axis=0)

        for i in range(len(OBSTACLES_XY)):
            OBSTACLES[i] = []
            for point in OBSTACLES_XY[i]:
                index = np.where(np.all(POINTS == point, axis=1))
                if len(index[0]) > 0:
                    OBSTACLES[i].append(index[0][0])

    except ValueError as e:
        print(f"ValueError: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function to check if a triangle is contained within any obstacle set
def is_triangle_in_obstacle(triangle, obstacle):
    return any(all(np.isin(triangle, obs)) for obs in obstacle)

def triangulate():
    global co,cf,triangulation,POINTS
    # Perform Delaunay triangulation
    triangulation = Delaunay(POINTS)

    # Loop through each triangle in 'space'
    for triangle in triangulation.simplices:
        # Check if the triangle is contained within any obstacle set
        if is_triangle_in_obstacle(triangle, OBSTACLES):
            # Append the triangle to 'co' if the condition is met
            co.append(triangle)
        else:
            # Append the triangle to 'cf' if the condition is not met
            cf.append(triangle)

    # Convert 'co' and 'cf' to numpy arrays
    co = np.array(co)
    cf = np.array(np.unique(cf,axis=0))

def display_graph(fig,frame):
    graph = FigureCanvasTkAgg(fig,master = frame)
    graph.get_tk_widget().pack(pady=5)
    toolbar = NavigationToolbar2Tk(graph,frame)
    toolbar.update()
    toolbar.pack(anchor='w',fill = tk.X)


def triangulation_window():
    global co,triangulation
    tri = tk.Toplevel()
    tri.iconbitmap("PSO_Logo2.ico")
    tri.title("Triangular Decomposition")
    tri_frame = tk.Frame(tri)
    tri_frame.grid(column=1,row=0)
    tri_label = tk.Label(tri_frame, text="Triangular Decomposition", font=("Unispace", 16))
    tri_label.pack(pady=10)
    # Plot the original points and the generated triangles
    fig, ax = plt.subplots(figsize=(9,9))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1],  color = 'darkorange')

    display_graph(fig,tri_frame)


def dijkstra():
    global shortest_path,G,cf,triangulation,sp
    for i in range(len(robots)):
        start_xy = robots[i]
        start_tri = (sorted(triangulation.simplices[triangulation.find_simplex(start_xy)]))
        goal_tri = (sorted(triangulation.simplices[triangulation.find_simplex(END_XY)]))

        # Add nodes (triangles) to the graph
        for simplex in cf:
            triangle_nodes = tuple(sorted(simplex))  # Sort the vertices to create a unique identifier
            G.add_node(triangle_nodes)

        # Calculate the distance (e.g., Euclidean distance) between triangles and add edges
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2 and len(set(node1).intersection(node2)) == 2:
                    triangle1 = triangulation.points[list(node1)]
                    triangle2 = triangulation.points[list(node2)]
                    distance = np.linalg.norm(triangle1.mean(axis=0) - triangle2.mean(axis=0))
                    G.add_edge(node1, node2, weight=distance)

        # Perform Dijkstra's algorithm to find the shortest path between two triangles
        # triangle input is backwards of output from simplexes
        start_node = (start_tri[0], start_tri[1], start_tri[2])  # Replace with your desired starting triangle
        end_node = (goal_tri[0], goal_tri[1], goal_tri[2])  # Replace with your desired ending triangle

        sp.append(nx.shortest_path(G, source=start_node, target=end_node, weight='weight'))

# Define objective function
def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def obj_function_distance(j,particles):
    global total_distance
    total_distance = distance(robots[j][0],[particles[0],particles[1]])
    for i in range(0, len(particles) - 3, 2):
        total_distance += distance((particles[i], particles[i + 1]), (particles[i + 2], particles[i + 3]))
    total_distance += distance(robots[j],[particles[0],particles[1]])
    return total_distance

def dijkstra_window():
    global G,shortest_path,co,midpoints,sp,colors
    dji = tk.Toplevel()
    dji.iconbitmap("PSO_Logo2.ico")
    dji.title("Dijkstra's Algorithm")
    dji_frame = tk.Frame(dji)
    dji_frame.grid(column=1,row=1,padx=10)
    dji_label = tk.Label(dji, text="Dijkstra's Algorithm", font=("Unispace", 16))
    dji_label.grid(column=1,row=0,columnspan=2,pady=10)

    # Calculate midpoints for all triangles
    triangle_midpoints = {}
    for triangle in G.nodes:
        if len(midpoints)==len(robots):
            break
        else:
            triangle_indices = np.array(triangle)
            triangle_points = triangulation.points[triangle_indices]
            edge_midpoints = calculate_edge_midpoints(triangle_points)
            triangle_midpoints[triangle] = edge_midpoints
        
    for i in range(len(robots)):
        if len(midpoints)==len(robots):
            break
        else:
            midpoints.append([])

            # Draw lines through midpoints
            for j in range(len(sp[i]) - 1):
                current_triangle = sp[i][j]
                next_triangle = sp[i][j+1]
                for element in triangle_midpoints[current_triangle]:
                    if any(np.array_equal(element,next_elem)for next_elem in triangle_midpoints[next_triangle]):
                        next_midpoints = element
                midpoints[i].extend(next_midpoints)

    # Plot the original points and the generated triangles
    fig, ax = plt.subplots(figsize=(9,9))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    for i in robots:
        # Annotate and plot 'START' in blue
        ax.plot(i[0], i[1], 'o', color=colors[robots.index(i)])
        ax.text(i[0], i[1], ' ROBOT '+str(robots.index(i)+1), verticalalignment='bottom', horizontalalignment='right', color=colors[robots.index(i)], fontweight='bold')


    # Annotate and plot 'END' in red
    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    # Draw lines connecting midpoints
    for midpoint in midpoints:
        for i in range(0,len(midpoint)-2,2):
            ax.plot([midpoint[i],midpoint[i+2]],
                    [midpoint[i+1],midpoint[i+3]],
                    color=colors[midpoints.index(midpoint)], linestyle='--')
            
        ax.plot([robots[midpoints.index(midpoint)][0],midpoint[0]],[robots[midpoints.index(midpoint)][1],midpoint[1]],color=colors[midpoints.index(midpoint)],linestyle='--')
        ax.plot([END_XY[0],midpoint[-2]],[END_XY[1],midpoint[-1]],color=colors[midpoints.index(midpoint)],linestyle='--')
            

    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

    display_graph(fig,dji_frame)

    data = [
    ["Robot",  "Djikstra's Algorithm\nDistance"],
    ]

    for i in range(len(robots)):
        j=i+1
        data.append(["ROBOT "+str(j),"{:.2f}".format(obj_function_distance(i,midpoints[i]))])

    data_frame = tk.Frame(dji)
    data_frame.grid(column=2,row=1,padx=10,pady=10)

    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            label = tk.Label(data_frame, text=cell,borderwidth=1, relief="solid",padx=10,pady
            =10,font=("Helvetica",16))
            label.grid(row=i, column=j, sticky="nsew")

# Define update methods

def get_next_velocity(v, c1, c2, gb, particles):
    return v + c1 + (c2 * (gb - particles))

def get_next_position(particles, velocity, lb, ub):
    return np.clip(particles + velocity, lb, ub)

def optimize():
    global shortest_path,gb,triangulation,sp,gbests,END_XY
    shortest_path_xy = []
    gbests = [[] for _ in range(len(robots))]
    gb = []

    for shortest_path in sp:
        # Change the shortest path array into a numpy array with the coordinates of the triangles instead of the points
        shortest_path_xy.append(triangulation.points[np.array(shortest_path)])
    
    for j in range(len(shortest_path_xy)):
        path_edges = []
        ub_x = []
        lb_x = []
        ub_y = []
        lb_y = []

        for i in range(len(shortest_path_xy[j]) - 1):
            current_triangle = shortest_path_xy[j][i]
            next_triangle = shortest_path_xy[j][i + 1]
            edge = []
            for element in current_triangle:
                if any(np.array_equal(element,next_elem)for next_elem in next_triangle):
                    edge.append(element)
            path_edges.append(edge)

        for i in range(len(path_edges)):
            ub_x.append(max(path_edges[i][0][0], path_edges[i][1][0]))
            lb_x.append(min(path_edges[i][0][0], path_edges[i][1][0]))
            ub_y.append(max(path_edges[i][0][1], path_edges[i][1][1]))
            lb_y.append(min(path_edges[i][0][1], path_edges[i][1][1]))

        # Combining lb_x and lb_y in alternating order starting from lb_x
        lb = np.array([val for pair in zip(lb_x, lb_y) for val in pair])

        # Combining ub_x and ub_y in alternating order starting from ub_x
        ub = np.array([val for pair in zip(ub_x, ub_y) for val in pair])

        d = len(sp[j]) - 1
        n = 2 * d
        particles = [np.random.uniform(lb, ub) for _ in range(n)]
        particles = np.array(particles)
        velocity = np.zeros_like(particles)
        n_iter = 1000
        c1 = np.zeros_like(particles)
        c2 = 0.5
        gb.append(particles[np.argmin(np.apply_along_axis(lambda e:obj_function_distance(j,e), 1, particles.reshape(-1, n)))])

        # This variation of PSO emphasizes global best

        # Main loop for PSO
        for i in range(n_iter):
            # Update global best
            current_min = np.argmin(np.apply_along_axis(lambda e:obj_function_distance(j,e), 1, particles.reshape(-1, n)))
            current_gbest = particles[current_min]
            gbests[j].append(current_gbest)
            if obj_function_distance(j,current_gbest) < obj_function_distance(j,gb[j]):
                gb[j]= current_gbest

            # Update weights, particle velocities and positions
            c1 = 0. * (pow(0.99, i)) * np.random.uniform(lb, ub)
            velocity = get_next_velocity(velocity, c1, c2, gb[j], particles)
            particles = get_next_position(particles, velocity, lb, ub)


def optimization_window():
    global co,shortest_path,POINTS,triangulation,start_xy,END_XY,gb,colors
    opt = tk.Toplevel()
    opt.iconbitmap("PSO_Logo2.ico")
    opt.title("Particle Swarm Optimization")
    opt_frame = tk.Frame(opt)
    opt_frame.grid(column=1,row=1,padx=10)
    opt_label = tk.Label(opt, text="Particle Swarm Optimization", font=("Unispace", 16))
    opt_label.grid(column=1,row=0,columnspan=2,pady=10)

    # Plot the original points and the generated triangles
    fig, ax = plt.subplots(figsize=(9,9))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    for i in robots:
        # Annotate and plot 'START' in blue
        ax.plot(i[0], i[1], 'o', color=colors[robots.index(i)])
        ax.text(i[0], i[1], ' ROBOT '+str(robots.index(i)+1), verticalalignment='bottom', horizontalalignment='right', color=colors[robots.index(i)], fontweight='bold')

    # Annotate and plot 'END' in red
    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    for gbs in range(len(gb)):
        # Draw lines connecting global best particles
        for i in range(0,len(gb[gbs])-2,2):
            ax.plot([gb[gbs][i],gb[gbs][i+2]],
                    [gb[gbs][i+1],gb[gbs][i+3]],
                    color=colors[gbs], linestyle='-')
            
    ax.plot([robots[gbs][0],gb[gbs][0]],[robots[gbs][1],gb[gbs][1]],color=colors[gbs],linestyle='-')
    ax.plot([END_XY[0],gb[gbs][-2]],[END_XY[1],gb[gbs][-1]],color=colors[gbs],linestyle='-')

    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

    display_graph(fig,opt_frame)

    data = [
    ["Robot",  "Particle Swarm Optimization\nDistance"],
    ]

    for i in range(len(robots)):
        j=i+1
        data.append(["ROBOT "+str(j),"{:.2f}".format(obj_function_distance(i,gb[i]))])

    data_frame = tk.Frame(opt)
    data_frame.grid(column=2,row=1,padx=10,pady=10)

    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            label = tk.Label(data_frame, text=cell,borderwidth=1, relief="solid",padx=10,pady
            =10,font=("Helvetica",16))
            label.grid(row=i, column=j, sticky="nsew")

    button_frame = tk.Frame(opt,padx=10,pady=10)
    button_frame.grid(column=3,row=1)
    label = tk.Label(button_frame,text="")


    for i in range(len(robots)):
        tk.Button(button_frame,text="ROBOT "+str(i+1)+" SIMULATION",command=lambda i=i:simulation_window(i),padx=10,pady=10).pack()



def simulation_window(robot):
    global gbests,co,shortest_path,robots
    plt.close('all')
    sim = tk.Toplevel()
    sim.iconbitmap("PSO_Logo2.ico")
    sim.title("Particle Swarm Optmization Simulation")
    sim_frame = tk.Frame(sim)
    sim_frame.grid(column=2,row=1,padx=10)
    gen = tk.Label(sim)
    gen.grid(column=1,row=0,columnspan=2,pady=10)
    fig,ax = plt.subplots(figsize=(9,9))

    for i in range(len(gbests[robot])):
        gen['text'] = "Iteration " + str(i)
        ax.clear()
        x_values = [gbests[robot][i][j] for j in range(0, len(gbests[robot][i]), 2)]
        y_values = [gbests[robot][i][j] for j in range(1, len(gbests[robot][i]), 2)]

        # Plot the original points and the generated triangles
        ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
        ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

        # Annotate and plot 'START' in blue
        ax.plot(robots[robot][0], robots[robot][1], 'o', color=colors[robot])
        ax.text(robots[robot][0], robots[robot][1], ' ROBOT '+str(robot+1), verticalalignment='bottom', horizontalalignment='right', color=colors[robot], fontweight='bold')

        # Annotate and plot 'END' in red
        ax.plot(END_XY[0], END_XY[1], 'o', color='red')
        ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

        # Highlight the triangles in the shortest path
        for triangle in shortest_path:
            triangle_indices = np.array(triangle)
            ax.fill(triangulation.points[triangle_indices, 0], triangulation.points[triangle_indices, 1], alpha=0.5, color='lightblue')

        for triangle in co:
            co_indices = np.array(triangle)
            ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

        # Plotting the current generation's best solutions
        ax.plot(x_values, y_values,'o')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)  # Adjust the pause time as needed for animation speed
   # Create a canvas in Tkinter and add the matplotlib figure to it
    canvas = FigureCanvasTkAgg(fig, master=sim_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    button = tk.Button(sim,text="CLOSE",command=sim.destroy())
    button.grid(column=3,row=1,fill=tk.BOTH, expand=True)


def comparison_window():
    global midpoints,gb,POINTS,start_xy,END_XY,shortest_path,co
    com = tk.Toplevel()
    com.iconbitmap("PSO_Logo2.ico")
    com.title("Comparison of Pure Dijkstra's Algorithm vs Particle Swarm Optimization")
    com_frame = tk.Frame(com)
    com_frame.grid(column=1,row=1,columnspan=2,padx=10)
    com_label = tk.Label(com, text="Comparison of Pure Dijkstra's Algorithm vs Particle Swarm Optimization", font=("Unispace", 16))
    com_label.grid(column=1,row=0,columnspan=2,pady=10)
    # Plot the original points and the generated triangles
    fig, ax = plt.subplots(figsize=(9,9))
    ax.triplot(POINTS[:, 0], POINTS[:, 1], triangulation.simplices.copy())
    ax.plot(POINTS[:, 0], POINTS[:, 1], 'o')

    # Calculate midpoints for all triangles
    triangle_midpoints = {}
    for triangle in G.nodes:
        if len(midpoints)==len(robots):
            break
        else:
            triangle_indices = np.array(triangle)
            triangle_points = triangulation.points[triangle_indices]
            edge_midpoints = calculate_edge_midpoints(triangle_points)
            triangle_midpoints[triangle] = edge_midpoints
            
    for i in range(len(robots)):
        if len(midpoints)==len(robots):
            break
        else:
             midpoints.append([])

            # Draw lines through midpoints
        for j in range(len(sp[i]) - 1):
            current_triangle = sp[i][j]
            next_triangle = sp[i][j+1]
            for element in triangle_midpoints[current_triangle]:
                if any(np.array_equal(element,next_elem)for next_elem in triangle_midpoints[next_triangle]):
                    next_midpoints = element
            midpoints[i].extend(next_midpoints)

    for i in robots:
        # Annotate and plot 'START' in blue
        ax.plot(i[0], i[1], 'o', color=colors[robots.index(i)])
        ax.text(i[0], i[1], ' ROBOT '+str(robots.index(i)+1), verticalalignment='bottom', horizontalalignment='right', color=colors[robots.index(i)], fontweight='bold')

    # Annotate and plot 'END' in red
    ax.plot(END_XY[0], END_XY[1], 'o', color='red')
    ax.text(END_XY[0], END_XY[1], ' END', verticalalignment='top', horizontalalignment='left', color='red', fontweight='bold')

    # Draw lines connecting midpoints
    for midpoint in midpoints:
        for i in range(0,len(midpoint)-2,2):
            ax.plot([midpoint[i],midpoint[i+2]],
                    [midpoint[i+1],midpoint[i+3]],
                    color=colors[midpoints.index(midpoint)], linestyle='--')
        ax.plot([robots[midpoints.index(midpoint)][0],midpoint[0]],[robots[midpoints.index(midpoint)][1],midpoint[1]],color=colors[midpoints.index(midpoint)],linestyle='--')
        ax.plot([END_XY[0],midpoint[-2]],[END_XY[1],midpoint[-1]],color=colors[midpoints.index(midpoint)],linestyle='--')

    for triangle in co:
        co_indices = np.array(triangle)
        ax.fill(triangulation.points[co_indices, 0], triangulation.points[co_indices, 1], color='darkorange')

    for gbs in range(len(gb)):

        # Draw lines connecting global best particles
        for i in range(0,len(gb[gbs])-2,2):
            ax.plot([gb[gbs][i],gb[gbs][i+2]],
                    [gb[gbs][i+1],gb[gbs][i+3]],
                    color=colors[gbs], linestyle='-',label='Dijkstra + PSO' if i == 0 else "")
        ax.plot([robots[gbs][0],gb[gbs][0]],[robots[gbs][1],gb[gbs][1]],color=colors[gbs],linestyle='-')
        ax.plot([END_XY[0],gb[gbs][-2]],[END_XY[1],gb[gbs][-1]],color=colors[gbs],linestyle='-')

    # Adding the legend outside the plot to the right
    ax.legend(loc='center right', bbox_to_anchor=(1, 1))

    display_graph(fig,com_frame)
    data = [
        ["Robot", "Djikstra Algorithm\nDistance", "Particle Swarm\nOptimization Distance","Distance Decreased","Percentage Decreased"],
    ]

    percentage = []
    for i in range(len(robots)):
        j=i+1
        percentage.append((obj_function_distance(i,midpoints[i])-obj_function_distance(i,gb[i]))/obj_function_distance(i,midpoints[i])*100)
        data.append(["ROBOT "+str(j),"{:.2f}".format(obj_function_distance(i,midpoints[i])),"{:.2f}".format(obj_function_distance(i,gb[i])),"{:.2f}".format(obj_function_distance(i,midpoints[i])-obj_function_distance(i,gb[i])),"{:.2f}%".format(percentage[i])])

    data_frame = tk.Frame(com)
    data_frame.grid(column=3,row=1,padx=10,pady=10)

    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            label = tk.Label(data_frame, text=cell,borderwidth=1, relief="solid",padx=10,pady
            =10,font=("Helvetica",16))
            label.grid(row=i, column=j, sticky="nsew")

    reduction_frame = tk.Frame(com)
    label1 = tk.Label(reduction_frame,text="Average distance reduction: ",font=("Helvetica",16))
    label2 = tk.Label(reduction_frame,text="{:.2f}%".format(sum(percentage)/len(percentage)),font=("Unispace",16),fg='green')
    reduction_frame.grid(column=3,row =0,padx=10,pady=10)
    label1.grid(column=0,row=0,padx=10,pady=10)
    label2.grid(column=1,row=0,padx=10,pady=10)

def simulate():
    initialize_inputs()
    triangulate()
    dijkstra()
    optimize()
    switch_off(simulate_button)
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

def draw_obstacle():
    global obstacle_points
    for i in range(len(obstacle_points)):
        x1, y1 = obstacle_points[i]
        x2, y2 = obstacle_points[(i+1) % len(obstacle_points)]
        canvas.create_line(x1, y1, x2, y2, fill="orange")
    obstacle_points = []

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
    for i in range(0, initial_map_size, 50):  # Draw grid lines every 50 units
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
root.title("Particle Swarm Optimization Simulator")
root.iconbitmap("PSO_Logo2.ico")

intro()

# Create two main frames: one for settings, one for simulation
settings_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

simulation_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
simulation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Label for Simulation Settings frame
settings_label = tk.Label(settings_frame, text="Simulation Settings", font=("Unispace", 16))
settings_label.pack(pady=10)

# Styling
btn_font = tkFont.Font(family="Unispace", size=12)
btn_padx = 20
btn_pady = 10
btn_relief = tk.FLAT
btn_bg = 'light grey'
btn_fg = 'black'

# Adding new disabled buttons at the top of the settings frame
triangulation_button = tk.Button(settings_frame, text="TRIANGULATION", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg, command=triangulation_window)
triangulation_button.pack(pady=5,padx=5,fill='x')

dijkstra_button = tk.Button(settings_frame, text="DIJKSTRA", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command = dijkstra_window)
dijkstra_button.pack(pady=5,padx=5,fill='x')

optimization_button = tk.Button(settings_frame, text="OPTIMIZATION", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command= optimization_window)
optimization_button.pack(pady=5,padx=5,fill='x')

comparison_button = tk.Button(settings_frame, text="COMPARISON", state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg,command=comparison_window)
comparison_button.pack(pady=5,padx=5,fill='x')

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
    canvas.update()
    draw_grid()  # Redraw the grid when the map size is updated

# Create buttons
button_frame = tk.Frame(settings_frame)
button_frame.pack(side=tk.LEFT, padx=10)


robot_button = tk.Button(button_frame, text="ADD ROBOT", command=set_robot, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
robot_button.pack(pady=5,padx=5,fill='x')

end_button = tk.Button(button_frame, text="ADD END POINT", command=set_end_point, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
end_button.pack(pady=5,padx=5,fill='x')

obstacle_button = tk.Button(button_frame, text="ADD OBSTACLE", command=set_add_obstacle, font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
obstacle_button.pack(pady=5,padx=5,fill='x')

# Label to display coordinates
coord_label = tk.Label(button_frame, text="", font=tkFont.Font(family="Helvetica", size=10))
coord_label.pack(pady=5)

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
    switch_off(triangulation_button)
    switch_off(dijkstra_button)
    switch_off(optimization_button)
    switch_off(comparison_button)
    initialization()

clear_button = tk.Button(button_frame, text="CLEAR", command=clear_canvas,state='disabled', font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg='grey', fg='white')
clear_button.pack(pady=5)

# Add the SIMULATE button
simulate_button = tk.Button(button_frame, text="SIMULATE", command=simulate,state="disabled", font=btn_font, padx=btn_padx, pady=btn_pady, relief=btn_relief, bg=btn_bg, fg=btn_fg)
simulate_button.pack(pady=10,padx=10,fill='x')

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
            coord_label.config(text="A maximum of 3 robots can be placed")
            return
        else:
            color = "blue"
            robots.append([x,initial_map_size-y])
            coord_label.config(text=str(current_mode).capitalize()+" "+str(len(robots))+" has been placed at "+coord_text+" !")
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
        if len(obstacle_points)==5:
            OBSTACLES_XY.append(np.array([[obstacle_points[0][0],initial_map_size- obstacle_points[0][1]],[obstacle_points[1][0],initial_map_size- obstacle_points[1][1]],[obstacle_points[2][0],initial_map_size- obstacle_points[2][1]],
                                          [obstacle_points[3][0],initial_map_size- obstacle_points[3][1]],[obstacle_points[4][0],initial_map_size- obstacle_points[4][1]]]))
            draw_obstacle()

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
canvas = tk.Canvas(simulation_frame, width=initial_map_size, height=initial_map_size, bg='white')
canvas.pack( )
canvas.bind("<Button-1>", on_canvas_click)
canvas.bind("<Motion>", show_tooltip)
canvas.bind("<Leave>", lambda e: hide_tooltip())


# Draw grid lines
draw_grid()

# Create a slider
slider_label = ttk.Label(root, text="Adjust Map Size:")
slider_label.pack()
slider = ttk.Scale(root, from_=10, to=900, orient="horizontal", command=update_map_size)
slider.set(initial_map_size)  # Set initial value
slider.pack(pady=10)

root.mainloop()
