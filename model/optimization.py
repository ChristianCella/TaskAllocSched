import pulp as lp
import numpy as np
import pandas as pd
import utilities.utils as ut
import data.data_handling as dh
import networkx as nx


# Function to count the number of tool changes from the current task to the end of the task matrix
# Analyzes the available tool for the task and count the total number of future tool changes 
# Returns the tool with the minimum number of tool changes

def count_tool_changes(vector_allocation, combined_df_sorted, available_tools, i):
    
    best_tool = None
    b_time = float('inf')
    end_effectors = combined_df_sorted.iloc[i]['END-EFFECTOR']
    tool_to_analyze = list(set(end_effectors.split(', ')))
    tool_changes_per_tool = np.zeros(len(tool_to_analyze))
    best_tool_changes = float('inf')
    time = np.zeros(len(tool_to_analyze))

    for tool in tool_to_analyze:

        selected_tool = tool
        tool_index = tool_to_analyze.index(tool)

        #Cycle through the tasks from the current task to the end of the task matrix
        for j in range(i+1, len(combined_df_sorted)):

            # Check if the task is assigned to the robot
            if vector_allocation[j] == 1:

                # Check if the selected tool is in the task available tools
                if (selected_tool not in combined_df_sorted.iloc[j]['END-EFFECTOR']):
                    # If the selected tool is not in the task available tools, count the tool change
                    tool_changes_per_tool[tool_index] += 1
                    # Check if the available tools are more than one
                    if len(combined_df_sorted.iloc[j]['END-EFFECTOR']) > 1:
                        # If the available tools are more than one, select the tool with the minimum number of future tool changes
                        selected_tool = count_tool_changes(vector_allocation, combined_df_sorted, available_tools, j)
                        time[tool_index] += combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                    else:
                        #If the available tools are only one, select the tool
                        selected_tool = combined_df_sorted.iloc[j]['END-EFFECTOR']
                        time[tool_index] += combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                else:
                    time[tool_index] += combined_df_sorted.iloc[j][f'tr_{selected_tool}_mounted']

        # Check if the current tool has the minimum number of tool changes
        if tool_changes_per_tool[tool_index] < best_tool_changes:
            best_tool = tool
            b_time = time[tool_index]
            best_tool_changes = tool_changes_per_tool[tool_index]

        # If the number of tool changes is the same, select the tool with the minimum time
        elif tool_changes_per_tool[tool_index] == best_tool_changes and time[tool_index] < b_time:
            best_tool = tool
            b_time = time[tool_index]
            best_tool_changes = tool_changes_per_tool[tool_index]

    
    return best_tool


# Branch and bound algorithm to find the best allocation

def branch_and_bound(division_constraint, initial_mounted_tool, available_tools ,RULA_AVERAGE_THRESHOLD, original_df, original_vector):

    n = len(original_df)

    best_allocation, best_time_end_tasks, final_robot_task, final_human_task, final_task_bb, final_task = None, None, None, None, None, None

    # Start with the best time end as infinity and the best RULA average as infinity
    best_max_time_end = float('inf')

    best_rula_average = float('inf') 

    # Create a list of shared tasks
    shared_tasks_columns = ['ID', 'time', 'tool', 'PRECEDENT', 'SHARED', 'Available', 'mount_tool']

    # The column 'Available' is a flag that indicates if the precedent task of a shared task is carried out
    # When the precedent one of a task is carried out, 'Available' = 1
    # We need it because the shared tasks can only be executed if both of their previous ones are carried out

    shared_tasks_list = []
    for i in range(n):
        if original_df.iloc[i]['SHARED'] != 0:
            shared_tasks_info = {
                'ID': original_df.iloc[i]['ID'],
                'time': 0,
                'tool': '0',
                'PRECEDENT': original_df.iloc[i]['PRECEDENT'],
                'SHARED': original_df.iloc[i]['SHARED'],
                'Available': False,
                'mount_tool': '0'
            }
            shared_tasks_list.append(shared_tasks_info)

    shared_tasks = pd.DataFrame(shared_tasks_list, columns=shared_tasks_columns)
    
    # Count the number of tasks not pre-assigned ('S' are the shared tasks not pre-assigned)
    r = original_vector.count('N') + original_vector.count('S')

    # Iterate over all possible allocations for not pre-assigned tasks
    for allocation in range(2**r):
        
        # Reset the vector to the original one
        vector = original_vector.copy()

        # Reset the combined_df_sorted and vector_allocation
        combined_df_sorted = original_df.copy()

        # Assign the flag to True for the tasks that have no precedent
        shared_tasks.loc[shared_tasks['PRECEDENT'] != 0, 'Available'] = False
        shared_tasks.loc[shared_tasks['PRECEDENT'] == 0, 'Available'] = True

        # Convert allocation number to binary string, we'll use this to insert pre-assigned task in the right position
        vector_allocation = [None] * n

        # Index for accessing bits in allocation_str
        allocation_index = 0
        allocation_str = f"{allocation:0{r}b}"

        rula_sum = 0

        # Insert pre-assigned tasks in the right position
        for i in range(n):
            if vector[i] == 'H':
                # Assign the task to the human
                vector_allocation[i] = 0
            elif vector[i] == 'R':
                # Assign the task to the robot
                vector_allocation[i] = 1
            else: 
                # Assign the task to the robot or human based on the allocation_str (the binary string of the allocation number that we generated)
                vector_allocation[i] = int(allocation_str[allocation_index])
                allocation_index += 1
            if vector_allocation[i] == 0:
                rula_sum += original_df.iloc[i]['rula']


        # Check if the division constraint is satisfied
        # If the division constraint is not satisfied, skip the current allocation to save time
        if (vector_allocation.count(1) / n) < division_constraint or (vector_allocation.count(0) / n) < division_constraint:

            continue

        #Check if the rula average constraint is satisfied. If it's not, skip the current allocation to save time
        #Check if there are not 's' in vector
        if 'S' not in vector:
            if rula_sum / vector_allocation.count(0) > RULA_AVERAGE_THRESHOLD:
                continue

        # Initialize the number of human and robot tasks, RULA sum, and other variables
        mounted_tool = initial_mounted_tool 
        rula_average, robot_end_time, human_end_time, start_time = 0, 0, 0, 0
        time_end_tasks = [0]*n

        # Create empty frames for human_task and robot_task 
        human_task = pd.DataFrame(columns=['ID', 'TASK', 'OBJECT', 'th', 'PRECEDENT', 'SHARED'])

        # Create empty frames for human_task and robot_task 
        robot_task = pd.DataFrame(columns=['ID','TASK', 'OBJECT', 'tr_gripper_mounted', 'tr_gripper_not_mounted', 'tr_vacuum_mounted','tr_vacuum_not_mounted', 'tr_magnetic_mounted', 'tr_magnetic_not_mounted','PRECEDENT', 'SHARED'])

        # Create an empty frame for the branch and bound table. We need it to store the tool and time used for each task
        task_bb = pd.DataFrame(columns=['ID','TASK', 'OBJECT', 'TOOL', 'tr', 'PRECEDENT', 'SHARED'])


        for i in range(n):
            
            # Extract the name of the task and its predecessor
            precedent_name = combined_df_sorted.iloc[i]['PRECEDENT']

            end_effectors = combined_df_sorted.iloc[i]['END-EFFECTOR']

            task = combined_df_sorted.iloc[i]

            #check if the ID of the task is not in the tables
            if task['ID'] not in human_task['ID'].values and task['ID'] not in robot_task['ID'].values and task['ID'] not in task_bb['ID'].values:

                # Manage NaN values in 'END-EFFECTOR' column
                if pd.isna(end_effectors) or not isinstance(end_effectors, str):
                    tools_in_task = set()
                else:
                    # Store the tools in the task as a set
                    tools_in_task = set(end_effectors.split(', '))

                # Check if the task in not shared
                if combined_df_sorted.iloc[i]['SHARED'] == 0:
                
                    if vector_allocation[i]==1:  # Task assigned to the robot

                        robot_task = dh.assign_to_robot(task, robot_task, available_tools)

                        # Tool selection based on the available tools and the tools in the task
                        # If the mounted tool is in the task, the robot will use it
                        if mounted_tool in tools_in_task:
                            time = combined_df_sorted.iloc[i][f'tr_{mounted_tool}_mounted']
                            tool = f'{mounted_tool} (mounted)'
                                
                        # If the mounted tool is not in the task, the robot will choose the tool with the minimum number of future tool changes
                        elif len(tools_in_task) > 1:
                            selected_tool = count_tool_changes(vector_allocation, combined_df_sorted, available_tools, i)
                            time = combined_df_sorted.iloc[i][f'tr_{selected_tool}_not_mounted']
                            tool = f'{selected_tool} (not mounted)'
                            mounted_tool = selected_tool

                        # If the task has only one tool, but different from the mounted_tool, the robot will use it
                        else:
                            selected_tool = combined_df_sorted.iloc[i]['END-EFFECTOR']
                            time = combined_df_sorted.iloc[i][f'tr_{selected_tool}_not_mounted']
                            tool = f'{selected_tool} (not mounted)'
                            mounted_tool = selected_tool
                            
                        # Here we calculate the finish time of the current task
                        # If the task has a precedent, we add the time to the precedent's finish time
                        if combined_df_sorted.iloc[i]['PRECEDENT'] != 0:

                            filtered_df = combined_df_sorted[combined_df_sorted['ID'] == precedent_name]

                            if not filtered_df.empty:
                                precedent_index = filtered_df.index[0]
                            
                                # Calculate the finish time of the current task by adding the time to the predecessor's finish time
                                time_end_tasks[i] = time_end_tasks[precedent_index] + time

                                # Take track of the robot end time
                                robot_end_time += time

                            # Check if the task's finish time is greater than the robot end time and vice versa
                            # We assign the maximum value to both
                            if (time_end_tasks[i] > robot_end_time):
                                robot_end_time = time_end_tasks[i]
                            else:
                                time_end_tasks[i] = robot_end_time

                        # If the task has no precedent, we check if is the first task assigned to the robot and human
                        #If is not the first taks, we check if is at least the first task assigned to the robot
                        elif(robot_end_time == 0):

                            # If is the first task, we assign the time to the task's finish time
                            time_end_tasks[i] = time
                            robot_end_time += time


                        #Last condition: the task has no precedent but is not the first task executed by the robot
                        #This is the case when we have more tasks assigned to the robot without precedent
                        else:

                            time_end_tasks[i] = robot_end_time + time  
                            robot_end_time += time  

                        task_bb = dh.add_task_to_bb(task, task_bb, tool, time) 


                    else:  # Task assigned to the human (same logic as the robot)

                        human_task = dh.assign_to_human(task, human_task)

                        time = combined_df_sorted.iloc[i]['th']

                        # Here we calculate the finish time of the current task
                        # If the task has a precedent, we add the time to the precedent's finish time
                        if combined_df_sorted.iloc[i]['PRECEDENT'] != 0:

                            filtered_df = combined_df_sorted[combined_df_sorted['ID'] == precedent_name]

                            if not filtered_df.empty:
                                precedent_index = filtered_df.index[0]
                            
                                # Calculate the finish time of the current task by adding the time to the predecessor's finish time
                                time_end_tasks[i] = time_end_tasks[precedent_index] + time

                                # Take track of the robot end time
                                human_end_time += time

                            # Check if the task's finish time is greater than the robot end time and vice versa
                            # We assign the maximum value to both
                            if (time_end_tasks[i] > human_end_time):
                                human_end_time = time_end_tasks[i]
                            else:
                                time_end_tasks[i] = human_end_time

                        # If the task has no precedent, we check if is the first task assigned to the robot and human
                        #If is not the first taks, we check if is at least the first task assigned to the robot
                        elif(human_end_time == 0):

                            # If is the first task, we assign the time to the task's finish time
                            time_end_tasks[i] = time
                            human_end_time += time

                        #Last condition: the task has no precedent but is not the first task executed by the robot
                        #This is the case when we have more tasks assigned to the robot without precedent
                        else:

                            time_end_tasks[i] = human_end_time + time  
                            human_end_time += time  

                    
                    #check if the task is in shared_precedent

                    for p in range(len(shared_tasks)):
                        if task['ID'] == shared_tasks.loc[p, 'PRECEDENT']:
                            shared_tasks.loc[p, 'Available'] = True

                # The task is shared
                else: 

                    task = combined_df_sorted.iloc[i]

                    # Extract the current task
                    shtask = shared_tasks.loc[shared_tasks['ID'] == task['ID']].iloc[0]

                    # Extract the shared task
                    sharedtask = shared_tasks.loc[shared_tasks['ID'] == shtask['SHARED']].iloc[0]

                    # Check if both tasks are available, it means that the precedent tasks are carried out
                    if shtask['Available'] == True and sharedtask['Available'] == True:
    
                        # Check if the task is not pre-assigned
                        if vector[i] == 'S':
                            
                            #Here we need to decide which task to assign to the robot and which to the human
                            #Save the index of the shared task
                            k = combined_df_sorted.index[combined_df_sorted['ID'] == task['SHARED']].tolist()
                            k = int(k[0])
                            
                            # Analyze the shared task to evaluate the time
                            # We need this for cycle to evaluate the time and the tool used for the taks assigned to the robot
                            for j in [i, k]:
                                
                                task = combined_df_sorted.iloc[j]
                                end_effectors = task['END-EFFECTOR']
                                tools_in_task = set(end_effectors.split(', '))

                                if mounted_tool in tools_in_task:
                                    time = combined_df_sorted.iloc[j][f'tr_{mounted_tool}_mounted']
                                    tool = f'{mounted_tool} (mounted)'

                                    
                                elif len(tools_in_task) > 1:
                                    selected_tool = count_tool_changes(vector_allocation, combined_df_sorted, available_tools, j)
                                    time = combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                                    tool = f'{selected_tool} (not mounted)'
                                    mounted_tool = selected_tool

                                else:
                                    selected_tool = combined_df_sorted.iloc[j]['END-EFFECTOR']
                                    time = combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                                    tool = f'{selected_tool} (not mounted)'
                                    mounted_tool = selected_tool

                                #Save the data in the shared_tasks table
                                shared_tasks.loc[shared_tasks['ID'] == task['ID'], 'time'] = time
                                shared_tasks.loc[shared_tasks['ID'] == task['ID'], 'tool'] = tool
                                shared_tasks.loc[shared_tasks['ID'] == task['ID'], 'mount_tool'] = mounted_tool
                            
                            #Save the ID of the tasks
                            id_i = combined_df_sorted.iloc[i]['ID']
                            id_k = int(combined_df_sorted.iloc[i]['SHARED'])

                            #Here we need to choose which allocation has the minimum time
                            attempt_1 = max (combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0])
                            attempt_2 = max (combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0])

                            if(attempt_1 < attempt_2):

                                #Assign the task i to the human
                                vector_allocation[i] = 0
                                vector[i] = 'H'
                                task = combined_df_sorted.iloc[i]
                                human_task = dh.assign_to_human(task, human_task)

                                #Assign the task k to the robot
                                vector_allocation[k] = 1
                                vector[k] = 'R'
                                task= combined_df_sorted.iloc[k]
                                robot_task = dh.assign_to_robot(task, robot_task, available_tools)

                                tool = shared_tasks.loc[shared_tasks['ID'] == id_k, 'tool'].values[0]
                                time = shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0]
                                mounted_tool = shared_tasks.loc[shared_tasks['ID'] == id_k, 'mount_tool'].values[0]
                                task_bb = dh.add_task_to_bb(task, task_bb, tool, time)

                            else:
                                vector_allocation[i] = 1

                                #Assign the task i to the robot
                                vector[i] = 'R'
                                task= combined_df_sorted.iloc[i]
                                robot_task = dh.assign_to_robot(task, robot_task, available_tools)

                                tool = shared_tasks.loc[shared_tasks['ID'] == id_i, 'tool'].values[0]
                                time = shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0]
                                mounted_tool = shared_tasks.loc[shared_tasks['ID'] == id_i, 'mount_tool'].values[0]
                                task_bb = dh.add_task_to_bb(task, task_bb, tool, time)

                                vector_allocation[k] = 0
                                #Assign the task k to the human
                                vector[k] = 'H'
                                task= combined_df_sorted.iloc[k]
                                human_task = dh.assign_to_human(task, human_task)
                            
                            #TIME

                            #Extract the task index that appears in the PRECEDENT column of shared tasks
                            precedent_i = combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'PRECEDENT'].values[0]
                            precedent_k = combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'PRECEDENT'].values[0]

                            precedent_index_i = combined_df_sorted.index[combined_df_sorted['ID'] == precedent_i].tolist()
                            precedent_index_k = combined_df_sorted.index[combined_df_sorted['ID'] == precedent_k].tolist()

                            if len(precedent_index_i) > 0:
                                precedent_index_i = int(precedent_index_i[0])
                            else: 
                                precedent_index_i = 0

                            if len(precedent_index_k) > 0:
                                precedent_index_k = int(precedent_index_k[0])
                            else:
                                precedent_index_k = 0

                            #Calculate the start time of the shared tasks

                            if precedent_index_i !=0 and precedent_index_k != 0:
                                #If both tasks have a precedent, we calculate the start time as the maximum between the end time of the robot, human, and the end time of the precedent tasks
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_i], time_end_tasks[precedent_index_k])
                            elif precedent_index_i == 0 and precedent_index_k != 0:
                                #If the task i has no precedent, we calculate the start time as the maximum between the end time of the robot, human, and the end time of the precedent task k
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_k])
                            elif precedent_index_i != 0 and precedent_index_k == 0:
                                #If the task k has no precedent, we calculate the start time as the maximum between the end time of the robot, human, and the end time of the precedent task i
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_i])
                            else:
                                #If both tasks have no precedent, we calculate the start time as the maximum between the end time of the robot and human
                                start_time = max(robot_end_time, human_end_time)

                            if vector_allocation[i] == 0:
                                #If the task i is assigned to the human, we calculate the end time as the maximum between the time of the task i and the time of the shared task k
                                time_end_tasks[i] = start_time + max(combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0])
                                time_end_tasks[k] = start_time + max(shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0], combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'th'].values[0])
                                #Update the robot and human end time
                                robot_end_time = time_end_tasks[k]
                                human_end_time = time_end_tasks[i]
                            else:
                                #If the task i is assigned to the robot, we calculate the end time as the maximum between the time of the task i and the time of the shared task k
                                time_end_tasks[i] = start_time + max(shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0], combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'th'].values[0])
                                time_end_tasks[k] = start_time + max(combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0])
                                #Update the robot and human end time
                                robot_end_time = time_end_tasks[i]
                                human_end_time = time_end_tasks[k]

                        # The task is pre-assigned, we assign the task to the robot or human based on the vector
                        else:
                            
                            #extract the task index against the combined_df.sorted table
                            k = combined_df_sorted.index[combined_df_sorted['ID'] == task['SHARED']].tolist()
                            k = int(k[0])
                            
                            # Same logic as before
                            for j in [i, k]:

                                task = combined_df_sorted.iloc[j]
                                
                                if vector_allocation[j] == 1:
                                
                                    end_effectors = task['END-EFFECTOR']
                                    if pd.notna(end_effectors) and isinstance(end_effectors, str):
                                        tools_in_task = set(end_effectors.split(', '))
                                    else:
                                        tools_in_task = set()

                                    if mounted_tool in tools_in_task:
                                        time = combined_df_sorted.iloc[j][f'tr_{mounted_tool}_mounted']
                                        tool = f'{mounted_tool} (mounted)'
                                        
                                    elif len(combined_df_sorted.iloc[j]['END-EFFECTOR']) > 1:
                                        selected_tool = count_tool_changes(vector_allocation, combined_df_sorted, available_tools, j)
                                        time = combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                                        tool = f'{selected_tool} (not mounted)'
                                        mounted_tool = selected_tool

                                    else:
                                        selected_tool = combined_df_sorted.iloc[j]['END-EFFECTOR']
                                        time = combined_df_sorted.iloc[j][f'tr_{selected_tool}_not_mounted']
                                        tool = f'{selected_tool} (not mounted)'
                                        mounted_tool = selected_tool
                                    
                                    task_bb = dh.add_task_to_bb(task, task_bb, tool, time)
                                    shared_tasks.loc[shared_tasks['ID'] == task['ID'], 'time'] = time
                                    shared_tasks.loc[shared_tasks['ID'] == task['ID'], 'tool'] = tool

                            
                            if vector_allocation[i] == 1:
                                task = combined_df_sorted.iloc[i]
                                robot_task = dh.assign_to_robot(task, robot_task, available_tools)                       
                                task = combined_df_sorted.iloc[k]
                                human_task = dh.assign_to_human(task, human_task)


                            else: 
                                task = combined_df_sorted.iloc[i]
                                human_task = dh.assign_to_human(task, human_task)
                                task = combined_df_sorted.iloc[k]
                                robot_task = dh.assign_to_robot(task, robot_task, available_tools)

                            id_i = combined_df_sorted.iloc[i]['ID']
                            id_k = int(combined_df_sorted.iloc[i]['SHARED'])


                            precedent_i = combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'PRECEDENT'].values[0]
                            precedent_k = combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'PRECEDENT'].values[0]
                            precedent_index_i = combined_df_sorted.index[combined_df_sorted['ID'] == precedent_i].tolist()
                            precedent_index_k = combined_df_sorted.index[combined_df_sorted['ID'] == precedent_k].tolist()

                            if len(precedent_index_i) > 0:
                                precedent_index_i = int(precedent_index_i[0])
                            else: 
                                precedent_index_i = 0

                            if len(precedent_index_k) > 0:
                                precedent_index_k = int(precedent_index_k[0])
                            else:
                                precedent_index_k = 0

                            if precedent_index_i !=0 and precedent_index_k != 0:
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_i], time_end_tasks[precedent_index_k])
                            elif precedent_index_i == 0 and precedent_index_k != 0:
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_k])
                            elif precedent_index_i != 0 and precedent_index_k == 0:
                                start_time = max(robot_end_time, human_end_time, time_end_tasks[precedent_index_i])
                            else:
                                start_time = max(robot_end_time, human_end_time)

                            if vector_allocation[i] == 0:
                                time_end_tasks[i] = start_time + max(combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0])
                                time_end_tasks[k] = start_time + max(shared_tasks.loc[shared_tasks['ID'] == id_k, 'time'].values[0], combined_df_sorted.loc[combined_df_sorted['ID'] == id_i, 'th'].values[0])                            
                                robot_end_time = time_end_tasks[k]
                                human_end_time = time_end_tasks[i]
                            else:
                                time_end_tasks[i] = start_time + max(shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0], combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'th'].values[0])
                                time_end_tasks[k] = start_time + max(combined_df_sorted.loc[combined_df_sorted['ID'] == id_k, 'th'].values[0], shared_tasks.loc[shared_tasks['ID'] == id_i, 'time'].values[0])
                                robot_end_time = time_end_tasks[i]
                                human_end_time = time_end_tasks[k]

                        #If the shared tasks are not consecutive, we need to arrange the tasks in the combined_df_sorted and vector_allocation
                        if (k > i+1):

                            # Scale tasks in combined_df_sorted and vector_allocation
                            temp = combined_df_sorted.iloc[k].copy()
                            temb_vect = vector_allocation[k]
                            temp_vect = vector[k]
                            temp_time = time_end_tasks[k]

                            for h in range (k-1, i, -1):
                                    
                                combined_df_sorted.iloc[h+1] = combined_df_sorted.iloc[h]
                                vector_allocation[h+1] = vector_allocation[h]
                                vector[h+1] = vector[h]
                                time_end_tasks[h+1] = time_end_tasks[h]

                            combined_df_sorted.iloc[i] = temp
                            vector_allocation[i] = temb_vect
                            vector[i] = temp_vect
                            time_end_tasks[i] = temp_time

                        elif (i > k+1):

                            temp = combined_df_sorted.iloc[k].copy()
                            temb_vect = vector_allocation[k]
                            temp_vect = vector[k]
                            temp_time = time_end_tasks[k]

                            for h in range (k+1, i):
                                    
                                combined_df_sorted.iloc[h-1] = combined_df_sorted.iloc[h]
                                vector_allocation[h-1] = vector_allocation[h]
                                vector[h-1] = vector[h]
                                time_end_tasks[h-1] = time_end_tasks[h]

                            combined_df_sorted.iloc[i] = temp
                            vector_allocation[i] = temb_vect
                            vector[i] = temp_vect
                            time_end_tasks[i] = temp_time
                            
                        #Impose flag of the shared tasks to False to avoid the repetition of the same task
                        shared_tasks.loc[shared_tasks['ID'] == id_i, 'Available'] = False
                        shared_tasks.loc[shared_tasks['ID'] == id_k, 'Available'] = False


            # To avoid useless iterations, we check if the end time of the robot and human is greater than the best maximum time end
            if (robot_end_time > best_max_time_end or human_end_time > best_max_time_end):
                break
                
        # Calculate the average RULA score and the maximum end time of the tasks
        max_time_end = max(time_end_tasks)

        if not human_task.empty:
            # Calculate the sum of the RULA scores of the tasks assigned to the human
            rula_sum = human_task['rula'].sum()

        human_tasks = len(human_task)

        if human_tasks > 0:
            # Calculate the average RULA score of the tasks assigned to the human
            rula_average = rula_sum / human_tasks

        # Convert the allocation vector to a binary number
        final_allocation = ut.vector_to_binary_number(vector_allocation)

        # Check if the average RULA score is below the threshold
        if rula_average <= RULA_AVERAGE_THRESHOLD:

            # Check if the current allocation is better than the best allocation found so far
            #If they are equal, we choose the one with the lowest average RULA score
            if max_time_end < best_max_time_end or (max_time_end == best_max_time_end and rula_average < best_rula_average):

                best_max_time_end = max_time_end
                best_allocation = final_allocation
                best_rula_average = rula_average
                best_time_end_tasks = time_end_tasks
                final_robot_task = robot_task
                final_human_task = human_task
                final_task_bb = task_bb
                final_task = combined_df_sorted

    return best_allocation, best_max_time_end, best_rula_average, best_time_end_tasks, final_robot_task, final_human_task, final_task_bb, final_task


# Function to filter tools in the 'END-EFFECTOR' column based on availability
def filter_available_tools(end_effectors, available_tools):
    # Split the string by commas and strip whitespace
    tools = end_effectors.split(', ')
    # Filter tools to include only those in the available list
    filtered_tools = [tool for tool in tools if tool in available_tools]
    # Join the filtered tools back into a string
    return ', '.join(filtered_tools)
    
# Function to order tasks while respecting the precedence constraint
# Tasks are ordered based on their priority, which is the number of direct and indirect descendant tasks
def order_tasks(df, G):

    # Calculate the dependency degree
    dependency_degree = {node: 0 for node in G.nodes}
    for node in G.nodes:
        # Add all direct and indirect descendants to the dependency degree
        dependency_degree[node] += len(nx.descendants(G, node))

    # Handle tasks that may be NaN or missing from dependency_degree
    for index, row in df.iterrows():
        task = row['TASK']
        if pd.notna(task) and task in dependency_degree:
            df.at[index, 'PRIORITY'] = dependency_degree[task]
        else:
            print(f"Missing or invalid task at index {index}: {task}")
            # Optionally set a default priority or handle the missing task appropriately
            df.at[index, 'PRIORITY'] = 0  # Set default priority if task is missing or NaN

    # SORTED

    # Create an empty table with the same structure as df
    sorted_df = pd.DataFrame(columns=df.columns)

    # Set the flag of the task with PRECEDENT equal to 0 to True
    df.loc[df['PRECEDENT'] == 0, 'Flag'] = True

    # Analyze the other tasks
    for index, row in df.iterrows():

        # Find the task with the highest priority among the tasks with flag = True
        max_priority = df[df['Flag'] == True]['PRIORITY'].max()
        # Select the task with the highest priority, if there are multiple, select only one
        task = df[(df['Flag'] == True) & (df['PRIORITY'] == max_priority)].head(1)
        #check if the task has shared ==0
        if task['SHARED'].values[0] == 0:
            # Add the task to the sorted_df table
            sorted_df = pd.concat([sorted_df, task])
            # Remove the task from the df table
            df.drop(task.index, inplace=True)
            # Set the flag of the tasks with PRECEDENT equal to the ID of the task just added to True
            df.loc[df['PRECEDENT'] == task['ID'].values[0], 'Flag'] = True
        else: 
            # Check if the shared task is already in the sorted_df
            if task['SHARED'].values[0] in sorted_df['ID'].values:
                # Add the task to the sorted_df table
                sorted_df = pd.concat([sorted_df, task])
                # Remove the task from the df table
                df.drop(task.index, inplace=True)
                # Set the flag of the tasks with PRECEDENT equal to the ID of the task just added to True
                df.loc[df['PRECEDENT'] == task['ID'].values[0], 'Flag'] = True
                #check if the shared task has flag == True

            elif df.loc[df['ID'] == task['SHARED'].values[0], 'Flag'].values[0] == True:
                # Add the task to the sorted_df table
                sorted_df = pd.concat([sorted_df, task])
                # Remove the task from the df table
                df.drop(task.index, inplace=True)
                # Set the flag of the tasks with PRECEDENT equal to the ID of the task just added to True
                df.loc[df['PRECEDENT'] == task['ID'].values[0], 'Flag'] = True
                # Set the priority of the shared task to 1000
                df.loc[df['ID'] == task['SHARED'].values[0], 'PRIORITY'] = 100

            else:
                #choose as task the one with gretaer priority but with shared == 0
                task = df[(df['Flag'] == True) & (df['PRIORITY'] == max_priority) & (df['SHARED'] == 0)].head(1)
                # Add the task to the sorted_df table
                sorted_df = pd.concat([sorted_df, task])
                # Remove the task from the df table
                df.drop(task.index, inplace=True)
                # Set the flag of the tasks with PRECEDENT equal to the ID of the task just added to True
                df.loc[df['PRECEDENT'] == task['ID'].values[0], 'Flag'] = True

    # Remove the Flag column
    sorted_df.drop('Flag', axis=1, inplace=True)
    #for a task with priority = 100 assigne the same priority of its shared task
    for index, row in sorted_df.iterrows():
        if row['PRIORITY'] == 100:
            sorted_df.at[index, 'PRIORITY'] = sorted_df.loc[sorted_df['ID'] == row['SHARED'], 'PRIORITY'].values[0]

    return sorted_df
    
# Function to create the MILP problem for task scheduling

def creation_milp_problem(all_tasks):

    schedule_prob = lp.LpProblem("Task_Scheduling", lp.LpMinimize)

    # Decisional variables (dictionaries)

    start_times = lp.LpVariable.dicts("start_time", (i for i in all_tasks.index), lowBound = 0, cat = 'Continuous')
    latest_end_time = lp.LpVariable("LatestEndTime", lowBound = 0, cat = 'Continuous')

    # Objective function

    weight_for_start_times = 0.01  # Weight for task start times to make a task start as soon as possible
    schedule_prob += latest_end_time + weight_for_start_times * lp.lpSum([start_times[i] for i in all_tasks.index])


    # Constraints: the task must end before the latest end time

    for i in all_tasks.index:
        schedule_prob += start_times[i] + all_tasks.at[i, 'Duration'] <= latest_end_time

    # Non - overlapping constraints: tasks assigned to the same agent can not be done at the same time

    for i in all_tasks.index: 
        for j in all_tasks.index:
            if i < j and all_tasks.at[i, 'Task_Type'] == all_tasks.at[j, 'Task_Type']:

                # A binary variable for each couple of tasks
                overlap_var = lp.LpVariable(f"Overlap_{i}_{j}", cat = 'Binary')
                schedule_prob += start_times[i] + all_tasks.at[i, 'Duration'] - start_times[j] <= 1e3 * (1 - overlap_var)
                schedule_prob += start_times[j] + all_tasks.at[j, 'Duration'] - start_times[i] <= 1e3 * overlap_var



    # Precedence constraints

    for i in all_tasks.index:
        # Check if there is a predecessor for the current task
        if all_tasks.at[i, 'PRECEDENT'] > 0:
            # 'PRECEDENT' should directly contain the ID of the predecessor task
            predecessor_id = all_tasks.at[i, 'PRECEDENT']
            # Find the predecessor task using the ID
            predecessor_list = all_tasks.index[all_tasks['ID'] == predecessor_id].tolist()
            
            if predecessor_list:  # Check if the list is not empty
                predecessor_index = predecessor_list[0]
                # Set the precedence constraint in the scheduling problem
                schedule_prob += start_times[i] >= start_times[predecessor_index] + all_tasks.at[predecessor_index, 'Duration']
            else:
                print(f"No predecessor task found for {all_tasks.at[i, 'TASK']} with precedence ID {predecessor_id}")



    # Constraints for shared tasks: they must start at the same time

    for i in all_tasks.index:
        if all_tasks.at[i, 'SHARED'] != 0:
            shared_id = all_tasks.at[i, 'SHARED']
            shared_index = all_tasks.index[all_tasks['ID'] == shared_id].tolist()
            
            if shared_index:
                shared_index = shared_index[0]
                # Check if both shared tasks have no predecessors
                if all_tasks.at[i, 'PRECEDENT'] == 0 and all_tasks.at[shared_index, 'PRECEDENT'] == 0:
                    # Set start times to zero
                    schedule_prob += start_times[i] == 0
                    schedule_prob += start_times[shared_index] == 0
                else:
                    # Apply synchronization constraint
                    schedule_prob += start_times[i] == start_times[shared_index], f"Sync_Start_{i}_{shared_index}"

    # Sequential order constraints for robot tasks based on the order in 'all_tasks' to handle tool changes
    for i in range(len(all_tasks) - 1):
       if all_tasks.at[i, 'Task_Type'] == 'Robot' and all_tasks.at[i + 1, 'Task_Type'] == 'Robot':
          # Forces the robot's task i+1 to start only after task i completes
         schedule_prob += start_times[i + 1] >= start_times[i] + all_tasks.at[i, 'Duration']

    #Tool Constraints for Robot Tasks in MILP Model:
    #First Task Constraint:
    #The first robot task has no tool constraints.
    #Subsequent Task Constraints:
    #For all subsequent robot tasks:
    #If a robot task's tool is specified as "x (mounted)":
    #The preceding robot task must use a tool with the name "x", which can be either "mounted" or "not mounted".
    #If a robot task's tool is specified as "x (not mounted)":
    #The preceding robot task must use a tool with the name "y", which can be either "mounted" or "not mounted", where "y" is different from "x".

    
    # Iterate over the robot tasks
    #for i in range(1, len(all_tasks)):
        #check if the task is a robot task and is not the first robot task
    #    if all_tasks.at[i, 'Task_Type'] == 'Robot':
            # Iterate over the previous tasks
     #       for j in range(i):
                #check if the previous task is a robot task
      #          if all_tasks.at[j, 'Task_Type'] == 'Robot':
                    # Check if the tool of the current task is "mounted"
       #             if 'not mounted' not in all_tasks.at[i, 'TOOL']:
                        # Check if the tool of the previous task is the same as the current task
         #               if all_tasks.at[j, 'TOOL'] == all_tasks.at[i, 'TOOL']:
                            # If the tool is the same, the current task can start only after the previous task ends (but we don't impose it)
          #                  schedule_prob += start_times[i] >= start_times[j] + all_tasks.at[j, 'Duration']
                            # if the tool is not the same, we need to impose that this task can't be execute after the previous task
           #             else:
            #                schedule_prob += start_times[i] >= start_times[j] + all_tasks.at[j, 'Duration'] + all_tasks.at[j, 'Duration']
                    # Check if the tool of the current task is "not mounted"
             #       else:
                        # Check if the tool of the previous task is different from the current task
              #          if all_tasks.at[j, 'TOOL'] != all_tasks.at[i, 'TOOL']:
                            # If the tools are different, the current task can start only after the previous task ends
               #             schedule_prob += start_times[i] >= start_times[j] + all_tasks.at[j, 'Duration']
                            # If the toola are the same, we need to impose that this task can't be execute after the previous task
                #        else:
                 #           schedule_prob += start_times[i] >= start_times[j] + all_tasks.at[j, 'Duration'] + all_tasks.at[j, 'Duration']

    return schedule_prob, start_times, latest_end_time