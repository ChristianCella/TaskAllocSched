import pandas as pd
import numpy as np
from config.settings import *

def create_initial_dataframes():
    """Crea i DataFrame iniziali usati nell'applicazione."""
    human_task = pd.DataFrame(columns=['ID', 'TASK', 'OBJECT', 'th', 'PRECEDENT', 'SHARED'])
    robot_task = pd.DataFrame(columns=['ID', 'TASK', 'OBJECT', 'tr_gripper_mounted', 'tr_gripper_not_mounted', 'tr_vacuum_mounted', 'tr_vacuum_not_mounted', 'tr_magnetic_mounted', 'tr_magnetic_not_mounted', 'PRECEDENT', 'SHARED'])
    task_bb = pd.DataFrame(columns=['ID', 'TASK', 'OBJECT', 'TOOL', 'tr', 'PRECEDENT', 'SHARED'])
    return human_task, robot_task, task_bb, 

# Function to generate a random allocation vector
""" 
Function to generate a vector of shared tasks (used in Layer 0.2 inside the function 'rand_task_matrix')
"""

def generate_shared_tasks(tasks):
    n = len(tasks)
    shared_dict = {i: "0" for i in range(n)}
    
    # If there are at least two tasks, we can assign shared tasks
    if n >= 2:
        attempts = 0
        while attempts < 100:
            attempts += 1
            task_indices = np.random.choice(n, size=2, replace=False)
            task_x, task_y = task_indices[0], task_indices[1]

            predecessor_x = tasks.iloc[task_x]['PRECEDENT']
            predecessor_y = tasks.iloc[task_y]['PRECEDENT']

            # Check that both tasks do not have PRECEDENT equal to '0'
            if predecessor_x == '0' or predecessor_y == '0':
                continue

            # Verify that both tasks have the same PRECEDENT; if so, assign them as shared tasks
            if predecessor_x == predecessor_y:
                shared_dict[task_x] = str(task_y + 1)
                shared_dict[task_y] = str(task_x + 1)
                break  

    shared_column = [shared_dict[i] for i in range(n)]
    return shared_column

# Function to assign a task to the robot

def assign_to_robot(task, robot_task, available_tools):

    task_data = {
        'ID': task['ID'],
        'TASK': task['TASK'],
        'OBJECT': task['OBJECT'],
        'PRECEDENT': task['PRECEDENT'],
        'END-EFFECTOR': task['END-EFFECTOR'],
        'SHARED': task['SHARED']
    }

    for tool in available_tools:
        mounted_col = f'tr_{tool}_mounted'
        not_mounted_col = f'tr_{tool}_not_mounted'
        
        task_data[mounted_col] = task.get(mounted_col, None)
        task_data[not_mounted_col] = task.get(not_mounted_col, None)
    
    new_task_df = pd.DataFrame([task_data])
    robot_task = pd.concat([robot_task, new_task_df], ignore_index=True)
    
    return robot_task


# Function to assign a task to the human

def assign_to_human (task, human_task):
                    
    human_task = pd.concat([human_task, pd.DataFrame({
    'ID': [task['ID']],
    'TASK': [task['TASK']],
    'OBJECT': [task['OBJECT']],
    'th': [task['th']],
    'PRECEDENT': [task['PRECEDENT']],
    'SHARED': [task['SHARED']],
    'rula': [task['rula']]
    })], ignore_index=True)

    return human_task

# Function to add a task to the branch and bound table

def add_task_to_bb(task, task_bb, tool, time):
    task_bb = pd.concat([task_bb, pd.DataFrame({
    'ID': [task['ID']],
    'TASK': [task['TASK']],
    'OBJECT': [task['OBJECT']],
    'TOOL': [tool],
    'tr': [time],
    'PRECEDENT': [task['PRECEDENT']],
    'SHARED': [task['SHARED']]
    })], ignore_index=True)

    return task_bb

# Function to randomly assign shared tasks

def assign_shared_tasks(shared_tasks ,human_shared_tasks, robot_shared_tasks, combined_df_sorted):

    if not human_shared_tasks.empty:

        # For each task assigned to a human, we assign the corresponding task to the robot
        for idx in human_shared_tasks.index:
            
            condition = (combined_df_sorted['ID'] == human_shared_tasks.at[idx, 'ID'])
            ind = np.where(condition)[0] 
            print (ind)
            combined_df_sorted.loc[ind, 'ASSIGNED'] = 'Human'
            shared_info = combined_df_sorted.at[ind[0], 'SHARED']

            if shared_info != 0:
                ind_x = combined_df_sorted.index[combined_df_sorted['ID'] == shared_info]   
                combined_df_sorted.at[ind_x[0], 'ASSIGNED'] = 'Robot'

                # Find the indices of the rows where 'ID' is equal to shared_task_id
                indices_to_drop = shared_tasks.index[shared_tasks['ID'] == shared_info].tolist()

                # Remove these rows from shared_tasks
                shared_tasks.drop(indices_to_drop, inplace=True)
            
            # Remove the task from the human_shared_tasks table
            human_shared_tasks.drop(index = idx, inplace = True)
        

    elif not robot_shared_tasks.empty:
        
        # For each task assigned to the robot, we assign the corresponding task to the human
        for idx in robot_shared_tasks.index:

            condition = (combined_df_sorted['ID'] == robot_shared_tasks.at[idx, 'ID'])
            ind = np.where(condition)[0] 

            combined_df_sorted.loc[ind, 'ASSIGNED'] = 'Robot'
            # Extract the ID of the shared task from the 'Shared' column
            shared_info = combined_df_sorted.at[ind[0], 'SHARED']

            if shared_info != 0:

                ind_x = combined_df_sorted.index[combined_df_sorted['ID'] == shared_info]   
                combined_df_sorted.at[ind_x[0], 'ASSIGNED'] = 'Human'

                # Find row indices where 'ID' equals shared_task_id
                indices_to_drop = shared_tasks.index[shared_tasks['ID'] == shared_info].tolist()

                # Remove these lines from shared_tasks
                shared_tasks.drop(indices_to_drop, inplace=True)
            
            # Let's remove the task from the human_shared_tasks table
            robot_shared_tasks.drop(index = idx, inplace = True)

    return combined_df_sorted, human_shared_tasks, robot_shared_tasks, shared_tasks


""" 
Function to generate random combinations of end effectors for each task 
(used in Layer 0.2, inside the function 'rand_task_matrix')
------------------------------------------------------------------------
* n = number of tasks
* options = list of available end-effectors
"""
def generate_end_effector_combinations(n, options):

    # Initialize a list to store the combinations
    combinations = []

    # Loop through all the tasks
    for _ in range(n):

        # Select a random number of end effectors to assign to each task
        num_eff = np.random.randint(1, len(options) + 1)

        # Choose a random combination with no repetitions (from 'options' choose 'size' elements, but put them back before choosing the next element)
        eff_comb = np.random.choice(options, size = num_eff, replace = False)

        # Join names with a comma to keep them in a single cell
        combinations.append(', '.join(eff_comb))

    return combinations

# Function to initialize tr values as None or NaN
def initialize_tr_columns(df, available_effectors):

    for eff in available_effectors:
        df[f'tr_{eff}_mounted'] = np.nan
        df[f'tr_{eff}_not_mounted'] = np.nan

# Function to assign tr values based on end-effectors
def assign_tr_values(row, available_tools, tool_change_time):

    # List of available end-effectors
    available_effectors = available_tools

    # Extract end-effector types from 'END-EFFECTOR' field and remove spaces
    effectors_in_task = {eff.strip() for eff in row['END-EFFECTOR'].split(',')}

    # Generate values for each mentioned effector

    for effector in available_effectors:
        if effector in effectors_in_task:
            mounted_value = np.random.uniform(low=3.0, high=8.0)
            # Round the mounted value to two decimal places
            row[f'tr_{effector}_mounted'] = round(mounted_value, 2)
            # Adds 2 seconds to the mounted value, also rounds this to two decimal places
            row[f'tr_{effector}_not_mounted'] = round(mounted_value + tool_change_time, 2)
        else:
            row[f'tr_{effector}_mounted'] = np.nan
            row[f'tr_{effector}_not_mounted'] = np.nan

    return row

""" 
Function to generate a random assembly DataFrame (Layer 0.1)
"""
def rand_gen_assembly (n, pos_x_lb, pos_x_ub, pos_y_lb, pos_y_ub, w_lb, w_ub, z_obj):

    # Create the columns for the assembly DataFrame (the tale variable is called 'assembly_df')
    object_names = [f"Obj_{i+1}" for i in range(n)]   
    positions = [(round(np.random.uniform(pos_x_lb, pos_x_ub), 2), round(np.random.uniform(pos_y_lb, pos_y_ub), 2), z_obj) for _ in range(n)]
    weights = np.random.randint(w_lb, w_ub, size = n)
    
    # The following values will be needed by the complexity based algorithm (e.g.: sensitivities[0] = 4 if obj_1 is robust, and so on ...)
    sensitivities = np.random.choice([0, 25, 50, 75, 100], size = n) # these are percentages
    dimensional_stabilities = np.random.choice([0, 25, 50, 75, 100], size = n)
    symmetry_values = np.random.choice([0, 50, 100], size = n)
    
    # Create the variable
    assembly_df = pd.DataFrame({
        "OBJECT": object_names,
        "POSITION": positions,
        "WEIGHT (g)": weights,
        "SENSITIVITY (0-4)": sensitivities,
        "DIMENSIONAL STABILITY (0-4)": dimensional_stabilities,
        "SYMMETRY (0-4)": symmetry_values
        })

    return assembly_df, positions, object_names

# Function to generate a random distance matrix

""" 
Function to generate a distance matrix (Layer 0.3)
---------------------------------------------------
* The position matrix created with 'rand_gen_position_matrix' (or 'gen_position_matrix') has 2 columns: 'PICK_(X,Y,Z)', 'PLACE_(X,Y,Z)'
* The human and robot positions are given as tuples (X,Y,Z)
* The distances are calculated as the Euclidean distance between two points in the 3D space
"""

def gen_distance_matrix(human_position, robot_position, positions, object_names, position_matrix):
    
    # Acquire the pick and place positions
    pick_positions = position_matrix['PICK_(X,Y,Z)']
    place_positions = position_matrix['PLACE_(X,Y,Z)']

    # calculate the distances
    distances_human_pick = [round(np.linalg.norm(np.array(pos) - np.array(human_position)),2) for pos in pick_positions]
    distances_human_place = [round(np.linalg.norm(np.array(pos) - np.array(human_position)),2) for pos in place_positions]
    distances_robot_pick = [round(np.linalg.norm(np.array(pos) - np.array(robot_position)),2) for pos in pick_positions]
    distance_robot_place = [round(np.linalg.norm(np.array(pos) - np.array(robot_position)),2) for pos in place_positions]

    # Creation of the Distance Matrix
    distance_matrix_df = pd.DataFrame({
        "OBJECT": object_names,
        "HUMAN_DISTANCE_PICK(mm)": distances_human_pick,
        "HUMAN_DISTANCE_PLACE(mm)": distances_human_place,
        "ROBOT_DISTANCE_PICK(mm)": distances_robot_pick,
        "ROBOT_DISTANCE_PLACE(mm)": distance_robot_place
    })
    
    return distance_matrix_df

"""
Random generation of the precedents 
(used in Layer 0.2 inside the function 'rand_task_matrix')
------------------------------------
* To keep track of the precedents, a dictionary is created
* The for loop is the key: it randomly generates the precedents ensuring there are no self-references or cycles
"""

def generate_precedents(tasks):
    precedents = {}

    # Generate precedents ensuring there are no self-references or cycles
    for i in tasks:
        
        # 'options' contains all elements from 1 to n, except the i-th       
        options = [0] + [x for x in tasks if x != i]
        to_remove = []
        
        # Loop over elements in 'options'
        for opt in options:
            
            # Impose the current equal to the opt-th element
            current = opt
            
            # While the current element is in the precedents dictionary
            while current in precedents:
                
                if precedents[current] == i:
                    
                    # Remove the opt-th element from the options list and break the loop
                    to_remove.append(opt)
                    break
                
                # else: assign the current-th element to the current variable
                current = precedents[current]
                
        # Remove the elements in 'to_remove' from 'options' 
        options = [x for x in options if x not in to_remove]

        # assign a random precedent among the remaining options or 0 if there are no options
        precedents[i] = np.random.choice(options) if options else 0

    # Convert the dictionary of precedents into a list for the DataFrame
    precedents_list = [precedents[i] for i in tasks]

    return precedents_list

""" 
Function to generate random data for the task matrix (Layer 0.2)
----------------------------------------------------------------
* Call the function to generate the end-effector combinations
* Call the function to generate the precedents
* Call the function to generate the shared tasks
* Check if the shared tasks have the same precedent and shared values
"""
def rand_task_matrix(n, task_names, object_names): 
    
    # Create random data for the task matrix
    mounting_directions = np.random.choice([0, 50, 100], size = n)
    mounting_tolerances = np.random.choice([0, 50, 100], size = n)
    mounting_resistance = np.random.choice([0, 50, 100], size = n)
    feeding_presentations = np.random.choice([0, 1, 3, 4], size = n)
    safety_tools = np.random.choice([0, 50, 100], size = n)
    safety_collisions = np.random.choice([0, 50, 100], size = n)
    fastening = np.random.choice([25, 50, 75], size = n)
    tasks = list(range(1, n + 1))
    end_effector_options = ["gripper", "vacuum", "magnetic", "screwdriver" ] # "Glue_Dispensing", "Welding_Torch"

    # Call teh function to generate the end-effector combinations
    end_effector_combinations = generate_end_effector_combinations(n, end_effector_options)

    # Generate random data for the applicability of the various phases
    mounting_applicable = np.random.choice([True, False], size = n)
    feeding_applicable = np.random.choice([True, False], size = n)
    fastening_applicable = np.random.choice([True, False], size = n)

    # Generate the list of precedents 
    precedents_list = generate_precedents(tasks) 

    # Definition of the 'task matrix'
    task_df = pd.DataFrame({
        "ID": tasks,
        "TASK": task_names,
        "OBJECT": object_names,  
        "MOUNTING_DIRECTION [0-2-4]": mounting_directions,
        "MOUNTING_TOLERANCE [0-2-4]": mounting_tolerances,
        "MOUNTING_RESISTANCE [0-2-4]": mounting_resistance,
        "FEEDING_PRESENTATION [0-1-3-4]" : feeding_presentations,
        "SAFETY_TOOLS [0-2-3-4]": safety_tools,
        "SAFETY_COLLISION (0-4)": safety_collisions,
        "FASTENING [1-2-3]": fastening,
        """ 
        These are needed to understand the characteristics of the task
        * example: is there something to be mounted to have a true HRC potential?
        """
        "MOUNTING_APPLICABLE": mounting_applicable,
        "FEEDING_APPLICABLE": feeding_applicable,
        "FASTENING_APPLICABLE": fastening_applicable,
        "PRECEDENT": precedents_list,
        "END-EFFECTOR": end_effector_combinations
    })

    # Add two additional columns: 'SHARED' and 'ASSIGNED'
    shared_column = generate_shared_tasks(task_df)
    shared_column = [int(x) for x in shared_column] 
    task_df['SHARED'] = shared_column
    task_df['ASSIGNED'] = 'None'

    """ 
    Solution of a problem: if a task has the same precedent and shared, generate a new list of precedents
    ------------------------------------------------------------------------------------------------------
    * The idea is to repeat this process a certain number of times (max_iterations) beacuse
        in case that flag_prec is raised, the precedents are randomly generated again and the check is repeated
    """
    max_iterations = 100

    for _ in range(max_iterations):

        #check if, for a specific task, the values in 'PRECEDENT' and 'SHARED' are equal
        flag_prec = False

        for i in range(n):
            if (task_df.loc[i, 'PRECEDENT'] != 0): # the first task has surely no precedent
                if task_df.loc[i, 'PRECEDENT'] == task_df.loc[i, 'SHARED']:

                    # Raise a flag
                    flag_prec = True
                    break

        # If the flag was raised: generate a new random list of precedents (This is very bad in reality)
        if flag_prec:

            new_precedent_list = generate_precedents()
            task_df['PRECEDENT'] = new_precedent_list
        else:

            # If no task has the same 'PRECEDENT' and 'SHARED' values, break the loop
            break

    return task_df

def rand_pre_simulation(task_df, robot_tasks_df, available_tools, tool_change_time, human_tasks_df):

    num_tasks = len(task_df)
    num_human_tasks = len(human_tasks_df)

    th = np.round(np.random.uniform(low = 4.0, high = 10.0, size = num_tasks), 2)

    # Initialize the tr columns in the DataFrame
    initialize_tr_columns(task_df, available_tools)
    initialize_tr_columns(robot_tasks_df, available_tools)

    # Applica la funzione per assegnare i valori di tr
    
    for index, row in task_df.iterrows():
        row = assign_tr_values(row, available_tools, tool_change_time)
        task_df.loc[index] = row

    for index, row in robot_tasks_df.iterrows():
        row = assign_tr_values(row, available_tools, tool_change_time)
        robot_tasks_df.loc[index] = row

    RulaScore = np.round(np.random.uniform(low = 1.0, high = 7.0, size = num_tasks), 2)

    # Update the columns in 'task_df'

    task_df = task_df.copy()
    task_df.loc[:, 'th'] = th
    task_df.loc[:, 'rula'] = RulaScore

    # Generate values for 'th' and 'RulaScore' based on 'num_human_tasks' (PRE-SIMULATION)

    th = np.round(np.random.uniform(low = 2.0, high = 10.0, size = num_human_tasks), 2)
    RulaScore = np.round(np.random.uniform(low = 1.0, high = 7.0, size = num_human_tasks), 2)

    # Update the values in the columns

    human_tasks_df['th'] = th
    human_tasks_df['rula'] = RulaScore

    return task_df, robot_tasks_df, human_tasks_df
    
def rand_pre_simulation_no_tool_change(task_df, robot_tasks_df, human_tasks_df):

    num_tasks = len(task_df)
    num_robot_tasks = len(robot_tasks_df)
    num_human_tasks = len(human_tasks_df)

    th = np.round(np.random.uniform(low = 1.0, high = 10.0, size = num_tasks), 2)
    tr = np.round(np.random.uniform(low = 1.0, high = 10.0, size = num_tasks), 2)
    RulaScore = np.round(np.random.uniform(low = 1.0, high = 7.0, size = num_tasks), 2)

    # Update the columns in 'task_df'
    task_df = task_df.copy()
    task_df.loc[:, 'th'] = th
    task_df.loc[:, 'rula'] = RulaScore
    task_df.loc[:, 'tr'] = tr

    th = np.round(np.random.uniform(low = 1.0, high = 10.0, size = num_human_tasks), 2)
    tr = np.round(np.random.uniform(low = 1.0, high = 10.0, size = num_robot_tasks), 2)
    RulaScore = np.round(np.random.uniform(low = 1.0, high = 7.0, size = num_human_tasks), 2)
    # Update the columns in 'human_tasks_df'
    human_tasks_df.loc[:, 'th'] = th
    human_tasks_df.loc[:, 'rula'] = RulaScore
    robot_tasks_df.loc[:, 'tr'] = tr

    return task_df, robot_tasks_df, human_tasks_df

""" 
Function to generate a non-random assembly DataFrame (Layer 0.1)
----------------------------------------------------------------
* the structure is the same as 'rand_gen_assembly'
"""
def gen_data ():

    # Hard-coded names (to be modified)
    object_names = ["Cube_1","Cube_2", "Cube_3", "Cube_4", "Cube_5", "Cube_6", "Cube_7", "Tray"]   
    positions = [(150,-400,0), (350,225,0), (500,400,0), (775,-550,0), (500,350,0), (225,-200,0), (225,275,0), (300,300,0)]
    weights = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    
    # The following values will be needed by the complexity based algorithm (hard-coded: to be changed)
    sensitivities = [100,100, 100, 100, 100, 100, 100, 100]
    dimensional_stabilities = [100, 100, 100, 100, 100, 100, 100, 100]
    symmetry_values = [100, 100, 100, 100, 100, 100, 100, 100]

    assembly_df = pd.DataFrame({
        "OBJECT": object_names,
        "POSITION (X,Y,Z)": positions,
        "WEIGHT (g)": weights,
        "SENSITIVITY (0-4)": sensitivities,
        "DIMENSIONAL STABILITY (0-4)": dimensional_stabilities,
        "SYMMETRY (0-4)": symmetry_values
        })

    task_names = ["PnP_Cube_1", "PnP_Cube_2", "PnP_Cube_3", "Cube4on1", "Cube5on2", "Cube6on3", "PnP_Cube_7", "PnStay_Tray"]

    return assembly_df, positions, object_names, task_names

def gen_task_matrix (n, task_names, object_names): 
    
    # Create random data for the task matrix

    mounting_directions = None
    mounting_tolerances = None
    mounting_resistance = None
    feeding_presentations = [100, 100, 100, 100, 100, 100, 100, 100]
    safety_tools = [100, 100, 100, 100, 100, 100, 100, 100]
    safety_collisions = [100, 100, 100, 100, 100, 100, 100, 100]
    fastening = None
    tasks = list(range(1, n + 1))

    # Generate random data for the applicability of the various phases

    mounting_applicable = [False, False, False, False, False, False, False, False]
    feeding_applicable = [True, True, True, True, True, True, True, True]
    fastening_applicable = [False, False, False, False, False, False, False, False]
    end_effector_combinations = ["gripper", "gripper", "gripper","gripper", "gripper", "gripper","gripper", " "]

    precedent_list = [0, 0, 0, 1, 2, 3, 1, 2]

    # Creation of the task matrix

    task_df = pd.DataFrame({
        "ID": tasks,
        "TASK": task_names,
        "OBJECT": object_names,  
        "MOUNTING_DIRECTION [0-2-4]": mounting_directions,
        "MOUNTING_TOLERANCE [0-2-4]": mounting_tolerances,
        "MOUNTING_RESISTANCE [0-2-4]": mounting_resistance,
        "FEEDING_PRESENTATION [0-1-3-4]" : feeding_presentations,
        "SAFETY_TOOLS [0-2-3-4]": safety_tools,
        "SAFETY_COLLISION (0-4)": safety_collisions,
        "FASTENING [1-2-3]": fastening,
        """ 
        These are needed to understand the characteristics of the task
        * example: is there something to be mounted to have a true HRC potential?
        """
        "MOUNTING_APPLICABLE": mounting_applicable,
        "FEEDING_APPLICABLE": feeding_applicable,
        "FASTENING_APPLICABLE": fastening_applicable,
        "PRECEDENT": precedent_list,
        "END-EFFECTOR": end_effector_combinations
    })

    # Convert the column "PRECEDENT" in integers

    task_df['ID'] = task_df['ID'].astype(int)
    task_df['PRECEDENT'] = task_df['PRECEDENT'].astype(int)
    shared_column = [0, 0, 0, 0, 0, 0, 8, 7]
    task_df['SHARED'] = shared_column
    task_df['ASSIGNED'] = 'None'

    return task_df
    
def pre_simulation(task_df, robot_tasks_df, human_tasks_df):


    # Crea la colonna th e rula a human task e task df, vuota per ora
    task_df['th'] = np.nan
    task_df['rula'] = np.nan
    human_tasks_df['th'] = np.nan
    human_tasks_df['rula'] = np.nan

    for tool in available_tools:

        mounted_col_name = f'tr_{tool}_mounted'
        not_mounted_col_name = f'tr_{tool}_not_mounted'
        
        # Inizializza le colonne con NaN
        robot_tasks_df[mounted_col_name] = np.nan
        robot_tasks_df[not_mounted_col_name] = np.nan

    th = [4.45, 2.91, 1.92, 2.15, 1.9, 4.15, 4.1, 4]

    tr_gripper_mounted = [7.45, 7.28, 6.82, 0, 7.15, 6.61, 0, 7.35]

    tr_gripper_not_mounted = [13.45, 13.28, 12.82, 6, 13.15, 12.61, 6, 13.35]

    rula = [3.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


    # Mappatura dei valori ai DataFrame in base all'ID

    human_tasks_df['th'] = human_tasks_df['ID'].apply(lambda x: th[x-1] if x <= len(th) else np.nan)
    human_tasks_df['rula'] = human_tasks_df['ID'].apply(lambda x: rula[x-1] if x <= len(rula) else np.nan)
    
    task_df['th'] = task_df['ID'].apply(lambda x: th[x-1] if x <= len(th) else np.nan)
    task_df['rula'] = task_df['ID'].apply(lambda x: rula[x-1] if x <= len(rula) else np.nan)

    
    task_df['tr_gripper_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    task_df['tr_gripper_not_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)
    task_df['tr_vacuum_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    task_df['tr_vacuum_not_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)
    task_df['tr_magnetic_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    task_df['tr_magnetic_not_mounted'] = task_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)
    

    robot_tasks_df['tr_gripper_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    robot_tasks_df['tr_gripper_not_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)
    robot_tasks_df['tr_vacuum_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    robot_tasks_df['tr_vacuum_not_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)
    robot_tasks_df['tr_magnetic_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_mounted[x-1] if x <= len(tr_gripper_mounted) else np.nan)
    robot_tasks_df['tr_magnetic_not_mounted'] = robot_tasks_df['ID'].apply(lambda x: tr_gripper_not_mounted[x-1] if x <= len(tr_gripper_not_mounted) else np.nan)

    return task_df, robot_tasks_df, human_tasks_df

""" 
Function to genrate the matrix for pick and place (Layer 0.3)
-------------------------------------------------------------
The structure is the same as 'rand_gen_position_matrix'

"""

def gen_position_matrix(assembly_df, positions, object_names):

    # Create the position matrix
    position_matrix = pd.DataFrame(columns=['TASK', 'PICK_(X,Y,Z)', 'PLACE_(X,Y,Z)'])

    # For each task, we will have two columns: one for the pick position and one for the place position
    for index, row in assembly_df.iterrows():
        task = row['ID']
        object_name = row['OBJECT']
        pick_position = assembly_df.loc[assembly_df['OBJECT'] == object_name, 'POSITION (X,Y,Z)'].values[0]
        place_position = [(round(np.random.uniform(150, 800), 2), round(np.random.uniform(0, 1100), 2), 0) for _ in range(n)]
        position_matrix = position_matrix.append({'TASK': task, 'PICK_(X,Y,Z)': pick_position, 'PLACE_(X,Y,Z)': place_position}, ignore_index=True)

    return position_matrix

""" 
Function to genrate the matrix for pick and place (Layer 0.3)
"""

def rand_gen_position_matrix(assembly_df, positions, object_names, pos_x_lb, pos_x_ub, pos_y_lb, pos_y_ub, w_lb, w_ub, z_obj):

    # Create the position matrix
    position_matrix = pd.DataFrame(columns=['TASK', 'PICK_(X,Y,Z)', 'PLACE_(X,Y,Z)'])

    # For each task, there are two columns: one for the pick position and one for the place position
    for index, row in assembly_df.iterrows():
        task = row['ID']
        object_name = row['OBJECT']
        pick_position = positions[index]

        # Generate a random place position
        place_position = (round(np.random.uniform(pos_x_lb, pos_x_ub), 2), round(np.random.uniform(pos_y_lb, pos_y_ub), 2), z_obj)

        # Create the matrix
        position_matrix = pd.concat([position_matrix, pd.DataFrame([{'TASK': task, 'PICK_(X,Y,Z)': pick_position, 'PLACE_(X,Y,Z)': place_position}])], ignore_index=True)

    return position_matrix