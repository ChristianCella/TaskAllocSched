import time
import numpy as np
import pandas as pd


# Definition of a method to calculate the weight score to the AUTOMATABILITY potential

def calculate_weight_score(weight):

    #If the weight is less than 1000 grams, the score is 100%
    #If the weight is between 1000 and 3000 grams, the score is 75%
    #If the weight is between 3000 and 5000 grams, the score is 50%
    #If the weight is between 5000 and 8000 grams, the score is 25%
    #If the weight is greater than 8000 grams, the score is 0%

    if weight > 8000:
        return 0
    elif 5000 < weight <= 8000:
        return 25
    elif 3000 < weight <= 5000:
        return 50
    elif 1000 < weight <= 3000:
        return 75
    else:  # weight <= 1000
        return 100    

# Definition of a method to calculate the distance score to the AUTOMATABILITY potential

def calculate_distance_score(distance):
    
    # PICK-PLACE DISTANCE
    #0% >100 cm 
    #25% 80-100cm
    #50% 60-80 cm 
    #75% 30-60 cm 
    #100% 0-30 cm


    if distance > 100:  
        return 0
    elif 80 < distance <= 100:
        return 25
    elif 60 < distance <= 80:
        return 50
    elif 30 < distance <= 60:
        return 75
    else:  # distance <= 30
        return 100
    
# Function to convert a vector of 0s and 1s to a binary number

def vector_to_binary_number(vector):
    # Convert the list of 0s and 1s to a string
    binary_string = ''.join(str(bit) for bit in vector)
    # Convert the binary string to a binary number
    binary_number = int(binary_string, 2)
    return binary_number

# Filter the 'END-EFFECTOR' column based on the available tools

# This code wotks under the assumption that at least one of the shared tasks is available for the robot
# Due the fact that we randomize the task and their characteristics, it is possible that both the shared task are not available for the robot
# If in the Layer1 we assigned a shared task to the human, we need to assign the other shared task to the robot
# But it can happen that the shared task assigned to the robot is not available for the robot for the Layer 2
# In this case, we need to assign to the robot at least one tool that is available for the task

def filter_end_effectors(combined_df_sorted, mounted_tool, available_tools):
    # Define a mask for rows where 'END-EFFECTOR' is NaN or empty
    mask = combined_df_sorted['END-EFFECTOR'].isnull() | (combined_df_sorted['END-EFFECTOR'] == '')

    # Set 'END-EFFECTOR' to the mounted_tool where the mask is True
    combined_df_sorted.loc[mask, 'END-EFFECTOR'] = mounted_tool

    # Set the times for mounted and not mounted tools where the mask is True
    combined_df_sorted.loc[mask, f'tr_{mounted_tool}_mounted'] = 6
    combined_df_sorted.loc[mask, f'tr_{mounted_tool}_not_mounted'] = 12

    for i in range(len(combined_df_sorted)):
        end_effector_string = combined_df_sorted.at[i, 'END-EFFECTOR']
        end_effector_list = end_effector_string.split(',')  # Split the string into a list based on the comma delimiter

        # Filter the list to only include tools that are in the available_tools list
        end_effector_list = [tool.strip() for tool in end_effector_list if tool.strip() in available_tools]

        # Join the list back into a string, if necessary
        combined_df_sorted.at[i, 'END-EFFECTOR'] = ', '.join(end_effector_list)


    return combined_df_sorted
            
    
# Function to check if there are duplicates in the combined_df_sorted, task_bb, and human_task dataframes

def check_duplicates(combined_df_sorted, task_bb, human_task):
    #Check if in the combined_df_sorted there are tasks with the same ID
    if combined_df_sorted['ID'].nunique() != len(combined_df_sorted):
        #eliminate one of the two tasks with the same ID
        combined_df_sorted = combined_df_sorted.drop_duplicates(subset = 'ID', keep = 'first')
        combined_df_sorted.reset_index(drop = True, inplace = True)
    #check also task_bb
    if task_bb['ID'].nunique() != len(task_bb):
        #eliminate one of the two tasks with the same ID
        task_bb = task_bb.drop_duplicates(subset = 'ID', keep = 'first')
        task_bb.reset_index(drop = True, inplace = True)
    #check also human_task
    if human_task['ID'].nunique() != len(human_task):
        #eliminate one of the two tasks with the same ID
        human_task = human_task.drop_duplicates(subset = 'ID', keep = 'first')
        human_task.reset_index(drop = True, inplace = True)

    return combined_df_sorted, task_bb, human_task

# Function to check if the shared tasks have the same duration

def assign_same_time_to_shared_tasks(all_tasks):
    
    for i in range(len(all_tasks)):

        if all_tasks.at[i, 'SHARED'] != 0:

            shared_indices = all_tasks.index[all_tasks['ID'] == all_tasks.at[i, 'SHARED']].tolist()

            ind_shared = shared_indices[0]

            if (all_tasks.at[i, 'Duration'] < all_tasks.at[ind_shared, 'Duration']):
                all_tasks.at[i, 'Duration'] = all_tasks.at[ind_shared, 'Duration']
            else:
                all_tasks.at[ind_shared, 'Duration'] = all_tasks.at[i, 'Duration']
    
    return all_tasks
