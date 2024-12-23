# Reset any existing timers or counters
timer_start = None

# Reset all global variables
try:
    # Collect the names of variables to delete
    vars_to_delete = [var for var in globals().copy() if var[0] != '_' and var not in ('np', 'pd', 'plt', 'lp', 'nx', 'timer', 'warnings')]
    for var in vars_to_delete:
        del globals()[var]
except NameError:
    pass

""" 
Code for the task allocation and scheduling
"""

import pandas as pd
from pandas.plotting import table

import numpy as np
import warnings
import pulp as lp
import time as timer
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

# Import the functions from the modules of this project
from config.settings import *
import data.data_handling as dh
import model.layers as ly
import model.optimization as opt
import model.matrix_update as matup
import utilities.utils as ut
import visualization.visualization as vis

# LaTeX commands and ignore warnings
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
warnings.filterwarnings("ignore")

# Set display options
pd.set_option('display.show_dimensions', False)

# Start the timer
start_count = timer.time()

""" 
Layer 0: Creation of random data for the objects
---------------------------------------
* the data are imported from config.settings.py
* The position is expressed with respect to a static reference system placed in the center of the table
* The table is 100 cm x 100 cm in size and the height of the objects is set to z_obj
* The weights are randomly generated between w_lb and w_ub grams
"""

print(f"\n\n")
print(f".....................................................LAYER 0.........................................................\n")

#########################################################################################
# Layer 0.1: Creation of the component matrix, position matrix and of the distance matrix
#########################################################################################

print(f"\n\n\n")
print(f"---------------------------Layer 0.1---------------------------\n")

if ifrandom:

    # Generate random data (that will be used for the distance matrix)
    assembly_df, positions, object_names = dh.rand_gen_assembly (n, pos_x_lb, pos_x_ub, pos_y_lb, pos_y_ub, w_lb, w_ub, z_obj)
    task_names = [f"Task_{i+1}" for i in range(n)]
else:

    # Generation of non-random data for the assembly matrix
    assembly_df, positions, object_names, task_names = dh.gen_data()

# Display the tables
if verbose:

    print("The table 'Component matrix' is the following:\n")
    print(assembly_df)


########################################
# Layer 0.2: Creation of the task matrix
########################################

print(f"\n\n\n")
print(f"---------------------------Layer 0.2---------------------------\n")

if ifrandom:

    # Random case: use the names generated above
    task_df = dh.rand_task_matrix(n, task_names, object_names)
else:

    # Non-random case: se the names obtained form the function 'gen_data'
    task_df = dh.gen_task_matrix(n, task_names, object_names)

# Display the table
if verbose:
    print("The table 'Task matrix' is the following\n")
    print(task_df)

############################################
# Layer 0.3: Creation of the Distance matrix
############################################

print(f"\n\n\n")
print(f"---------------------------Layer 0.3---------------------------\n")

# Generate the position matrix
if ifrandom:

    # Random case: use the names generated above
    position_matrix = dh.rand_gen_position_matrix(task_df, positions, object_names, pos_x_lb, pos_x_ub, pos_y_lb, pos_y_ub, w_lb, w_ub, z_obj)   
else:

    # Non-random case: se the names obtained form the function 'gen_data'
    position_matrix = dh.gen_position_matrix(task_df, positions, object_names)

# Generate the distance matrix
distance_matrix_df = dh.gen_distance_matrix(human_position, robot_position, positions, object_names, position_matrix)

if verbose:
    print("The 'Position matrix' is the following\n")
    print(position_matrix)

    print("\n\nThe 'Distance matrix' is the following\n")
    print(distance_matrix_df)

######################################################
# Layer 0.4: Creation of the precedence graph
#-----------------------------------------------------
# Tasks without any precedent are highlighted in black
######################################################

# Intantiate the DiGraph object
G = nx.DiGraph()

# Call the function to plot the precedence graph
vis.plot_precedence_graph(task_df, G, precedence_graph_plot)

# Display the DataFrame

if verbose:
    print("A reduced version of the 'Task matrix' is the following:\n")  
    print(task_df[['ID', 'TASK', 'PRECEDENT', 'SHARED']])

###########################################
# Layer 0.5: Assignment of the forced tasks
###########################################

# Get subsets of columns
human_tasks_df = pd.DataFrame(columns = ['ID', 'TASK', 'OBJECT', 'PRECEDENT', 'SHARED'])
robot_tasks_df = pd.DataFrame(columns = ['ID', 'TASK', 'OBJECT', 'PRECEDENT', 'END-EFFECTOR', 'SHARED'])

# Call the function to get the modified tables
task_df, human_tasks_df, robot_tasks_df = ly.Layer_0_5(task_df, human_tasks_df, robot_tasks_df)

"""
Layer 1: Evaluate the maximum load and distance for the robot
* Tasks are assigned to humans if weight and distance are greater than the robot's maximum load and distance
"""

# Call the function to assign tasks to humans based on the maximum load and distance
task_df, human_tasks_df, robot_tasks_df = ly.Layer_1(task_df, assembly_df, distance_matrix_df, robot_max_load, robot_max_distance, human_tasks_df, robot_tasks_df, position_matrix)

# Display the updated matrices

if verbose:
    print("\nUpdated matrix of tasks after LAYER 1:\n")
    print(task_df)
    print("\nUpdated matrix for humans after LAYER 1:\n")
    print(human_tasks_df)
    print("\nUpdated matrix for robots after LAYER 1:\n")
    print(robot_tasks_df)

"""
Layer 2: Evaluation of available tools 
* Tasks are assigned to the human if the available tools are not suitable for the task 
* If there are no tools available, the task is only feasible for the human
""" 

task_df, human_tasks_df, robot_tasks_df= ly.Layer_2(task_df, available_tools, mounted_tool, tool_change_feasible, human_tasks_df, robot_tasks_df)

# Filter the 'END-EFFECTOR' column based on the available tools

task_df['END-EFFECTOR'] = task_df['END-EFFECTOR'].apply(lambda x: opt.filter_available_tools(x, available_tools))

if verbose:
    print(f"\nUpdated task matrix after LAYER 2:\n{task_df}\n")
    print(f"\nUpdated Human task matrix after LAYER 2:\n{human_tasks_df}\n")

    
"""
Layer 3: Evaluation of the Automatability Potential 
* Tasks are assigned to the human if the Automatability Potential is below the threshold
* The 'Automatability Potential' indicates how feasible a task is for a robot


task_df, human_tasks_df, robot_tasks_df = ly.Layer_3(task_df, assembly_df, distance_matrix_df, human_tasks_df, robot_tasks_df, AUTOMATABILITY_THRESHOLD)

# Display the results

if verbose:
    
    print(f'\nUpdated task matrix after LAYER 3:\n{task_df}\n')
    print(f'\nUpdated Human task matrix after LAYER 3:\n{human_tasks_df}\n')
    print(f'\nUpdated Robot task matrix after LAYER 3:\n{robot_tasks_df}\n')

"""

# Reduce the task matrix to the columns of interest

task_df = task_df[['ID','TASK', 'OBJECT','PRECEDENT','END-EFFECTOR', 'SHARED']]


"""
PRE-SIMULATION
* We add to the task matrix the time and RULA score for each task
* Randomly generate the values
"""

""" 
Insert th and RulaScore in the matrix 'human_tasks_df'
"""

# Number of tasks assigned to the humans

num_human_tasks = len(human_tasks_df)

#Number of tasks assigned to the robot

num_robot_tasks = len(robot_tasks_df)

# Determine the number of rows of 'task_df'

num_tasks = len(task_df)

""" 
Insert th, tr and RulaScore in the matrix 'human_df'
"""

# Generate values for 'th', 'tr', and 'RulaScore' based on 'num_tasks'
if ifrandom:
    if tool_change_feasible:

        task_df, robot_tasks_df,human_tasks_df = dh.rand_pre_simulation(task_df, robot_tasks_df, available_tools, tool_change_time, human_tasks_df)

    else: 

        task_df, robot_tasks_df, human_tasks_df = dh.rand_pre_simulation_no_tool_change(task_df, robot_tasks_df, human_tasks_df)


    if verbose:
        print(f'\nUpdated task matrix after the PRE-SIMULATION:\n{task_df}\n')
        print(f'\nUpdated Human task matrix after the PRE-SIMULATION:\n{human_tasks_df}\n')
        print(f'\nUpdated Robot task matrix after the PRE-SIMULATION:\n{robot_tasks_df}\n')

else:

    task_df, robot_tasks_df, human_tasks_df = dh.pre_simulation(task_df, robot_tasks_df, human_tasks_df)

task_df.reset_index(drop = True, inplace = True)

print(f'\nTask matrix after the PRE-SIMULATION:\n{task_df}\n')
print(f'\nHuman task matrix after the PRE-SIMULATION:\n{human_tasks_df}\n')
print(f'\nRobot task matrix after the PRE-SIMULATION:\n{robot_tasks_df}\n')


"""
Layer 4: Evaluate the RULA above the threshold
* Tasks are assigned to robot if the RULA score is greater than 5
"""

task_df, human_tasks_df, robot_tasks_df = ly.Layer_4(task_df, human_tasks_df, RULA_THRESHOLD, tool_change_feasible, robot_tasks_df, available_tools)


if verbose:
    
    # Display the updated DataFrames
    print(f'\nUpdated task matrix after LAYER 4:\n{task_df}')
    print(f'\nUpdated Human task matrix after LAYER 4:\n{human_tasks_df}')
    print(f'\nUpdated Robot task matrix after LAYER 4:\n{robot_tasks_df}')


# Add the 'ASSIGNED' column to each table
task_df['ASSIGNED'] = 'None'
human_tasks_df['ASSIGNED'] = 'Human'
robot_tasks_df['ASSIGNED'] = 'Robot'


"""
Layer 5: Evaluate the risk and gain of the task
"""

task_df, human_tasks_df, robot_tasks_df = ly.Layer_5(task_df, end_table, position_matrix, human_tasks_df, robot_tasks_df)

human_tasks_df['ASSIGNED'] = 'Human'
robot_tasks_df['ASSIGNED'] = 'Robot'

if verbose:
        
    # Display the updated DataFrames
    print(f'\nUpdated task matrix after LAYER 5:\n{task_df}')
    print(f'\nUpdated Human task matrix after LAYER 5:\n{human_tasks_df}')
    print(f'\nUpdated Robot task matrix after LAYER 5:\n{robot_tasks_df}')


# Combine the tables
combined_df_sorted = pd.concat([task_df, human_tasks_df, robot_tasks_df], ignore_index=True)

combined_df_sorted.reset_index(drop=True, inplace=True)

# Apply the filter to the 'END-EFFECTOR' column
combined_df_sorted = ut.filter_end_effectors(combined_df_sorted, mounted_tool, available_tools)

#check if a task with tool == mounted tool and tr_gripper_mounted == 0 is present
for i in range(len(combined_df_sorted)):    
    if combined_df_sorted.at[i, 'END-EFFECTOR'] == mounted_tool and combined_df_sorted.at[i, f'tr_{mounted_tool}_mounted'] == 0:
        combined_df_sorted.at[i, f'tr_{mounted_tool}_mounted'] = 8
        combined_df_sorted.at[i, f'tr_{mounted_tool}_not_mounted'] = 14

"""
Task Allocation: branch-and-bound algorithm
* Expected output: the best complete task assignment to humans and robots
* All tasks need to be assigned and no branches are left to be explored 
"""

# Run the branch-and-bound algorithm defined above

best_allocation = None
num_human_tasks = len(human_tasks_df)
num_tasks_in_df = len(combined_df_sorted)  
num_robot_tasks = len(robot_tasks_df)
division_constraint_in = division_constraint
rula_tot = human_tasks_df['rula'].sum()
if num_human_tasks > 0:
    rula_average = round(rula_tot / num_human_tasks,2)
else:
    rula_average = 0


# Function to order tasks while respecting the precedence constraint
# Tasks are ordered based on their priority, which is the number of direct and indirect descendant tasks
combined_df_sorted = opt.order_tasks(combined_df_sorted, G)

#Reset the index
combined_df_sorted.reset_index(drop = True, inplace = True)

print(f'\nCombined matrices after ordering:\n{combined_df_sorted}\n')

# We need this to compare the different columns
combined_df_sorted['ID'] = combined_df_sorted['ID'].astype(int)
combined_df_sorted['PRECEDENT'] = combined_df_sorted['PRECEDENT'].astype(int)
combined_df_sorted['SHARED'] = combined_df_sorted['SHARED'].astype(int)

# Create the vector based on the 'Assigned' column
vector = ['N' if x == 'None' else 'H' if x == 'Human' else 'R' for x in combined_df_sorted['ASSIGNED']]

for i in range(0, len(combined_df_sorted)):
    if (combined_df_sorted.at[i, 'ASSIGNED'] == 'None' and combined_df_sorted.at[i, 'SHARED'] != 0):
        vector[i] = 'S'

if verbose:
    print("\nVector of assigned task:", vector)
    print(f'\nAverage RULA score before B&B: {rula_average}')

#Save the original combined_df_sorted
combined_df_sorted_original = combined_df_sorted.copy()
#Save the original vector
vector_original = vector.copy()

# Loop until it find a valid best_allocation or until it run out of constraints. 
# If it doesn't find a valid solution, it reduces the constraint and try again

while best_allocation is None and RULA_AVERAGE_THRESHOLD <= 7 :

    # Reset the division_constraint to the original value
    division_constraint_in = division_constraint

    while best_allocation is None and division_constraint_in > 0:

        # Try to find a best_allocation with the current constraint
        best_allocation, best_time, best_rula, time_end_tasks, robot_task, human_task, task_bb, combined_df_sorted = opt.branch_and_bound(division_constraint_in, mounted_tool, available_tools, RULA_AVERAGE_THRESHOLD, combined_df_sorted_original, vector_original)

        # If it doesn't find a valid solution, it reduce the constraint on the allocation division and try again

        if best_allocation is None:
            division_constraint_in -= 0.1
            if division_constraint_in > 0:
                print(f'\nDivision constraint reduced to: {round(division_constraint_in,2)}')
        else:
            break  
    
    # If it doesn't find a valid solution, it reduce the constraint on the RULA score and try again
    if best_allocation is None:
        RULA_AVERAGE_THRESHOLD += 0.5
        print(f'\nRULA constraint increased to: {RULA_AVERAGE_THRESHOLD}')
    else:
        break

# Convert the best allocation to a binary string
binary_allocation = format(best_allocation, f'0{num_tasks_in_df}b')

# Calculate the percentage of human and robot work
human_work = binary_allocation.count('0')
robot_work = binary_allocation.count('1')
human_work_percentage = human_work / num_tasks_in_df * 100
robot_work_percentage = robot_work / num_tasks_in_df * 100

# Calculate the number of tool changes
num_tool_changes = task_bb['TOOL'].str.count('(not mounted)').sum()

# Round each item to two decimal places
time_end_tasks_rounded = [round(time, 2) for time in time_end_tasks]

#Check if there are duplicates
combined_df_sorted, task_bb, human_task = ut.check_duplicates(combined_df_sorted, task_bb, human_task)

# Print the results
print("\nOptimal allocation (0 = Human, 1 = Robot):", binary_allocation)
print("\nBest total time:", round(best_time,2),"seconds")
print("\nHuman tasks:", human_work, f"({human_work_percentage:.2f}%)")
print("\nRobot tasks:", robot_work, f"({robot_work_percentage:.2f}%)")
print("\nAverage RULA score:", round(best_rula,2))
print(f'\nNumber of tool changes: {num_tool_changes}')
print("\nTime end matrix", time_end_tasks_rounded)
print(f'\nFinal Human Task Table: \n{human_task}\n')
print(f'\nRobot Task assigned in B&B: \n{task_bb}\n')
print(f'\nCombined matrix after B&B: \n{combined_df_sorted}\n')


"""
Task Scheduling: Mixed-Integer Linear Programming (MILP)
* The allocation is over ==> Human and Robot know what they have to do
* Now, the expected output of the scheduling is to determine when they do it
"""

# Add the column 'duration' to 'human_task' and 'robot_task' before the concatenation

human_task_with_duration = human_task.assign(Duration = human_task['th']).drop(['th'], axis = 1)
robot_task_with_duration = task_bb.assign(Duration = task_bb['tr']).drop(['tr'], axis = 1)


# Join the two matrices

all_tasks = pd.concat([human_task_with_duration.assign(Task_Type = 'Human'),
                       robot_task_with_duration.assign(Task_Type = 'Robot')]).reset_index(drop = True)


# Assign the same time to shared tasks

all_tasks = ut.assign_same_time_to_shared_tasks(all_tasks)

print(f'\nThe complete matrix all_tasks is:\n {all_tasks}')

# Create the scheduling problem
schedule_prob, start_times, latest_end_time = opt.creation_milp_problem(all_tasks)

""" 
Resolution of the problem
"""

schedule_prob.solve(lp.GLPK(path = "C:/Users/chris/Downloads/glpk_installation_folder/glpk-4.65/w64/glpsol.exe"))
# schedule_prob.solve(lp.GLPK(msg=False, path="C:\\Program Files\\glpk-4.65\\w64\\glpsol.exe"))

# Create a DataFrame with task names and start times
task_start_times = pd.DataFrame({
    'TASK': [all_tasks.at[i, 'TASK'] for i in all_tasks.index],
    'START_TIME': [start_times[i].value() for i in all_tasks.index]
})

# Sort the DataFrame by 'START_TIME'
task_start_times = task_start_times.sort_values(by='START_TIME')

# Print the sorted results
print("\nTask scheduling results (ordered by start time):\n")
for index, row in task_start_times.iterrows():
    print(f"{row['TASK']}: Starts at {row['START_TIME']}")

# Print the total execution time
print(f"\nThe total execution time is: {latest_end_time.value()} seconds\n")

fig, ax = plt.subplots(figsize=(10, 6))

# Build the data for the Gantt chart
task_data = []
tool_change_markers = []
for i, task in all_tasks.iterrows():
    start = start_times[i].varValue
    duration = task['Duration']
    tool_info = None
    # Color and type based on task type and tool info
    if task['Task_Type'] == 'Human':
        color = 'skyblue'
    else:
        tool_info = task_bb.loc[task_bb['TASK'] == task['TASK'], 'TOOL'].values[0]
        if 'gripper' in tool_info:
            color = 'lightgreen'
        elif 'vacuum' in tool_info:
            color = 'yellow'
        elif 'magnetic' in tool_info:
            color = 'orange'
        else:
            color = 'grey'  # Default color if no tool info found
    # Set edgecolor for shared tasks
    shared = task['SHARED']
    edgecolor = 'purple' if shared != 0 and shared != task['ID'] else color
    linewidth = 2 if shared != 0 and shared != task['ID'] else 0
    # Append the task info to task_data
    task_data.append(((start, duration), 10 * i, color, edgecolor, linewidth))
        # Identifica i task con tool "not mounted" e aggiungi un marker per il cambio di strumento
    if tool_info and 'not mounted' in tool_info:
        tool_change_duration = 3
        tool_change_start = start # Inizio del cambio di strumento
        tool_change_y_offset = 10 * i   # Centrato verticalmente (aggiustare +1 a seconda della necessità)
        tool_change_markers.append((tool_change_start, tool_change_duration, tool_change_y_offset))

# Add each bar to the chart
for i, (start_duration, y_offset, color, edgecolor, linewidth) in enumerate(task_data):
    start, duration = start_duration
    ax.broken_barh([(start, duration)], (y_offset, 9), facecolors=color, edgecolor=edgecolor, linewidth=linewidth)

# Aggiungi i marker per il cambio di strumento con il nuovo colore
for start, duration, y_offset in tool_change_markers:
    ax.broken_barh([(start, duration)], (y_offset, 9), facecolors='#8B0000')  # Altezza minore e centrato


# Axis settings

ax.set_yticks([10 * i + 5 for i in range(len(all_tasks))])
ax.set_yticklabels(all_tasks['TASK'])
ax.set_xlabel("Time [s]", fontsize=15)
ax.legend(handles=[
    Patch(facecolor='skyblue', label='Human'),
    Patch(facecolor='lightgreen', label='Robot (Gripper)'), 
    Patch(facecolor='yellow', label='Robot (Vacuum)'), 
    Patch(facecolor='orange', label='Robot (Magnetic)'),
    Patch(facecolor='white', edgecolor='purple', label='Shared Task'),
    Patch(facecolor='#8B0000', label='Tool Change') 
])

ax.set_title('Gantt Chart - Allocation & Scheduling', fontsize=20)

if update_matrix:
    
    # Initialize the time bands in the human task matrix
    human_task = matup.initialize_time_bands(human_task)

    # Generate random real times for tasks based on existing data
    real_human_times = matup.generate_random_times_based_on_existing(human_task)

    # Remove the original 'th' column
    human_task.drop(columns='th', inplace=True)

    # Print the complete matrix of random real times for tasks based on existing data
    if verbose:
        print("\nComplete matrix of random real times for tasks based on existing data:\n")
        print(real_human_times)

    # Rename the 'ID' column to 'Task_ID' for consistency
    human_task.rename(columns={'ID': 'Task_ID'}, inplace=True)

    # Rename the 'ID' column to 'Task_ID' for consistency
    real_human_times.rename(columns={'id': 'Task_ID'}, inplace=True)

    # Reshape the human task times to match the structure of the real times
    simulated_times = matup.reshape_human_task_times(human_task)

    # Apply the function to aggregate real times

    aggregated_real_times = matup.aggregate_real_times(real_human_times, 'Time')

    
    # Evaluate the weights based on the sensitivity analysis

    weights_df = matup.sensitivity_analysis(simulated_times, aggregated_real_times)

    # Call the function to update the human task matrix with weighted times
    updated_times = matup.update_human_tasks_with_weighted_times(simulated_times, weights_df, aggregated_real_times, human_task)

        
    # Round the updated times to two decimal places
    updated_times[['th_0-3h', 'th_3-6h', 'th_6h+']] = updated_times[['th_0-3h', 'th_3-6h', 'th_6h+']].round(2)

    # Print the updated human task matrix
    print("\nUpdated human task matrix with weighted times:\n")
    print(updated_times)

# Stop the timer
end_count = timer.time()
 
# Calculate the elapsed time
elapsed_time = end_count - start_count

print(f"\nTotal execution time of the algorithm: {elapsed_time:.3f} seconds")

# Display the Gantt chart
plt.grid(True)
plt.show()