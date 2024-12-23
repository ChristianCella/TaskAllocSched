import pandas as pd
import numpy as np
import data.data_handling as dh
import utilities.utils as ut

""" 
Function used in Layer 0.5 to assign tasks to humans and robots.
----------------------------------------------------------------
* It modifies the 'task_df' DataFrame by removing tasks that require specific needs
"""

def Layer_0_5(task_df, human_tasks_df, robot_tasks_df):

    # Empty variables to store the indices of the tasks to remove
    indices_to_eliminate = []

    for i in range(len(task_df)):
        
        if i not in indices_to_eliminate:

            # Verify if the task must be assigned to the human
            if task_df.at[i, 'ASSIGNED'] == 'Human':

                human_tasks_df = human_tasks_df.append(task_df.loc[i])
                indices_to_eliminate.append(i)

                if task_df.at[i, 'SHARED'] != 0:

                    #assign the task that have the ID = SHARED to the robot, since the other task is assigned to the human
                    robot_tasks_df = robot_tasks_df.append(task_df.loc[task_df['ID'] == task_df.at[i, 'SHARED']])
                    indices_to_eliminate.append(task_df[task_df['ID'] == task_df.at[i, 'SHARED']].index[0])

            # Verify if the task must be assigned to the robot
            elif task_df.at[i, 'ASSIGNED'] == 'Robot':

                robot_tasks_df = robot_tasks_df.append(task_df.loc[i])
                indices_to_eliminate.append(i)

                if task_df.at[i, 'SHARED'] != 0:

                    #assign the task that have the ID = SHARED to the human
                    human_tasks_df = human_tasks_df.append(task_df.loc[task_df['ID'] == task_df.at[i, 'SHARED']])
                    indices_to_eliminate.append(task_df[task_df['ID'] == task_df.at[i, 'SHARED']].index[0])

    # Drop the rows that have been assigned to humans or robots (if not, they will be assigned in the next layers)
    task_df.drop(indices_to_eliminate, inplace = True)
    task_df.reset_index(drop = True, inplace = True)

    return task_df, human_tasks_df, robot_tasks_df

def Layer_4(task_df, human_tasks_df, RULA_THRESHOLD, tool_change_feasible, robot_tasks_df, available_tools):

    indices_to_drop = []

    if tool_change_feasible==False:

        for index in reversed(task_df.index):

            # Verify if the task should be assigned to the robot

            if task_df.at[index, 'rula'] >=  RULA_THRESHOLD:
                
                task = task_df.loc[index]
                
                robot_tasks_df = dh.assign_to_robot (task_df.loc[index], robot_tasks_df)

                if task_df.at[index, 'SHARED'] != 0:

                    id_shared = task_df.at[index, 'SHARED'].astype(int)
                    shared_row = task_df.loc[task_df['ID'] == id_shared]
                    new_row_shared = pd.DataFrame({'ID':[shared_row['ID'].iloc[0]],'TASK': [shared_row['TASK'].iloc[0]], 'OBJECT': [shared_row['OBJECT'].iloc[0]], 'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]], 'SHARED': [shared_row['SHARED'].iloc[0]], 'th': [shared_row['th'].iloc[0]], 'rula': [shared_row['rula'].iloc[0]]})
                    human_tasks_df = pd.concat([human_tasks_df, new_row_shared], ignore_index = True)
                    indices_to_drop.append(shared_row.index[0])

                # Remove the task from 'task_df'
                
                indices_to_drop.append(index)

    else: 
        
        robot_tasks_df = robot_tasks_df[['ID','TASK', 'OBJECT', 'tr_gripper_mounted', 'tr_gripper_not_mounted', 'tr_vacuum_mounted','tr_vacuum_not_mounted', 'tr_magnetic_mounted', 'tr_magnetic_not_mounted', 'END-EFFECTOR','PRECEDENT', 'SHARED']]

        for index, row in task_df.iterrows():

            # Verify if the task should be assigned to the robot

            if row['rula'] >= RULA_THRESHOLD:
                
                task = task_df.loc[index]
                if index not in indices_to_drop:
                    robot_tasks_df = dh.assign_to_robot (task, robot_tasks_df,available_tools)

                if task_df.at[index, 'SHARED'] != 0:

                    id_shared = int(task_df.at[index, 'SHARED'])
                    shared_row = task_df.loc[task_df['ID'] == id_shared]
                   
                    if not shared_row.empty:
                        new_row_shared = pd.DataFrame({'ID':[shared_row['ID'].iloc[0]],'TASK': [shared_row['TASK'].iloc[0]], 'OBJECT': [shared_row['OBJECT'].iloc[0]], 'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]], 'SHARED': [shared_row['SHARED'].iloc[0]], 'th': [shared_row['th'].iloc[0]], 'rula': [shared_row['rula'].iloc[0]]})
                        
                        if shared_row.index[0] not in indices_to_drop:
                            human_tasks_df = pd.concat([human_tasks_df, new_row_shared], ignore_index = True)
                            indices_to_drop.append(shared_row.index[0])

                if index not in indices_to_drop:
                    indices_to_drop.append(index)
                
    # Reset the indices after the modifications

    task_df.drop(indices_to_drop, inplace=True)
    task_df.reset_index(drop = True, inplace = True)
    robot_tasks_df.reset_index(drop = True, inplace = True)
    human_tasks_df.reset_index(drop = True, inplace = True)

    return task_df, human_tasks_df, robot_tasks_df


def Layer_1(task_df, assembly_df, distance_matrix_df, robot_max_load, robot_max_distance, human_tasks_df, robot_tasks_df, position_matrix):
    
    # Create the matrix of the tasks assigned to humans

    indices_to_remove = set() 

    for index, row in task_df.iterrows():

        obj_name = row['OBJECT']

        # Verify if the task is already assigned to a human or robot
        if row['ID'] in human_tasks_df['ID'].values or row['ID'] in robot_tasks_df['ID'].values or index in indices_to_remove:

            continue

        elif distance_matrix_df.loc[distance_matrix_df['OBJECT'] == obj_name, 'ROBOT_DISTANCE_PICK(mm)'].values[0] > robot_max_distance or distance_matrix_df.loc[distance_matrix_df['OBJECT'] == obj_name, 'ROBOT_DISTANCE_PLACE(mm)'].values[0] > robot_max_distance:
            
            # Create a new row and concatenate it to 'human_tasks_df'; also remove them from 'task_df'
            new_row = pd.DataFrame({'ID':row['ID'],'TASK': [row['TASK']], 'OBJECT': [obj_name], 'PRECEDENT': [row['PRECEDENT']], 'SHARED': [row['SHARED']]})
            #check if new_row shared is equal to any id id human task df
            if row['ID'] not in human_tasks_df['ID'].values and row['ID'] not in robot_tasks_df['ID'].values:
              
                human_tasks_df = pd.concat([human_tasks_df, new_row], ignore_index = True)
                indices_to_remove.add(index)

                # If a shared task is assigned to the human, the shared task is assigned to the human
                if new_row['SHARED'].values[0] != 0:
                    id_shared = new_row['SHARED'].values[0]
                    shared_row = task_df.loc[task_df['ID'] == id_shared]
                    if not shared_row.empty:
                        new_row_shared = pd.DataFrame({
                            'ID': [shared_row['ID'].iloc[0]],
                            'TASK': [shared_row['TASK'].iloc[0]],
                            'OBJECT': [shared_row['OBJECT'].iloc[0]],
                            'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]],
                            'END-EFFECTOR': [shared_row['END-EFFECTOR'].iloc[0]],
                            'SHARED': [shared_row['SHARED'].iloc[0]]
                        })
                        if new_row_shared['ID'].values[0] not in human_tasks_df['ID'].values and new_row_shared['ID'].values[0] not in robot_tasks_df['ID'].values:
                            robot_tasks_df = pd.concat([robot_tasks_df, new_row_shared], ignore_index=True)
                            indices_to_remove.update(shared_row.index.tolist())
            
        # The condition on distance is not met: verify now the weight    

        elif assembly_df.loc[assembly_df['OBJECT'] == obj_name, 'WEIGHT (g)'].values[0] > robot_max_load:
            
            # Create a new row and concatenate it to 'human_tasks_df'; also remove them from 'task_df'
            new_row_df = pd.DataFrame([{'ID':row['ID'],'TASK': row['TASK'], 'OBJECT': obj_name, 'PRECEDENT': row['PRECEDENT'], 'SHARED': row['SHARED']}])  
            
            if row['ID'] not in human_tasks_df['ID'].values and row['ID'] not in robot_tasks_df['ID'].values:

                human_tasks_df = pd.concat([human_tasks_df, new_row_df], ignore_index = True)
                indices_to_remove.add(index)

                # If a shared task is assigned to the human, the shared task is assigned to the human
                if new_row_df['SHARED'].values[0] != 0:
                    id_shared = new_row_df['SHARED'].values[0]
                    shared_row = task_df.loc[task_df['ID'] == id_shared]
                    if not shared_row.empty:
                        new_row_shared = pd.DataFrame({
                            'ID': [shared_row['ID'].iloc[0]],
                            'TASK': [shared_row['TASK'].iloc[0]],
                            'OBJECT': [shared_row['OBJECT'].iloc[0]],
                            'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]],
                            'END-EFFECTOR': [shared_row['END-EFFECTOR'].iloc[0]],
                            'SHARED': [shared_row['SHARED'].iloc[0]]
                        })
                        if new_row_shared['ID'].iloc[0] not in human_tasks_df['ID'].values and new_row_shared['ID'].iloc not in robot_tasks_df['ID'].values:
                            robot_tasks_df = pd.concat([robot_tasks_df, new_row_shared], ignore_index = True)
                            indices_to_remove.update(shared_row.index.tolist())

    # Reset the indices after some rows were removed
    # Rimuovi tutte le righe identificate in una sola volta
    task_df.drop(list(indices_to_remove), inplace=True)
    task_df.reset_index(drop=True, inplace=True)
    human_tasks_df.reset_index(drop = True, inplace = True)
    robot_tasks_df.reset_index(drop = True, inplace = True)

    return task_df, human_tasks_df, robot_tasks_df


def Layer_2(task_df, available_tools, mounted_tool, tool_change_feasible, human_tasks_df, robot_tasks_df):

    indices_to_remove = set()

    if tool_change_feasible:

        # Iterate over task_df and assign tasks based on tool availability

        for index in reversed(task_df.index):

            #check if the index is in the indices to remove
            if index in indices_to_remove:

                continue

            else:

                # Get the list of tools needed for the current task
                tools_in_task = set(task_df.at[index, 'END-EFFECTOR'].split(', '))

                # Check if available tools intersect with needed tools
                if not tools_in_task or not tools_in_task.intersection(available_tools):

                    if task_df.at[index, 'ID'] not in human_tasks_df['ID'] and task_df.at[index, 'ID'] not in robot_tasks_df['ID']:

                        # Create a temporary DataFrame with the task to add to human_tasks_df
                        task_to_add = pd.DataFrame({
                            'ID': [int(task_df.at[index, 'ID'])],
                            'TASK': [task_df.at[index, 'TASK']],
                            'OBJECT': [task_df.at[index, 'OBJECT']],
                            'PRECEDENT': [task_df.at[index, 'PRECEDENT']],
                            'SHARED': [task_df.at[index, 'SHARED']]
                        })

                        # Filter columns that are completely empty or contain only NA
                        cols_to_use = task_to_add.columns[task_to_add.notna().any()]
                        
                        human_tasks_df = pd.concat([human_tasks_df, task_to_add[cols_to_use]], ignore_index=True)
                                        
                        if task_df.at[index, 'SHARED'] != 0:

                            id_shared = task_df.at[index, 'SHARED'].astype(int)
                            shared_row = task_df.loc[task_df['ID'] == id_shared]
                            new_row_shared = pd.DataFrame({
                                'ID': [shared_row['ID'].iloc[0]],
                                'TASK': [shared_row['TASK'].iloc[0]],
                                'OBJECT': [shared_row['OBJECT'].iloc[0]],
                                'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]],
                                'END-EFFECTOR': [shared_row['END-EFFECTOR'].iloc[0]],
                                'SHARED': [shared_row['SHARED'].iloc[0]]
                            })
                            if new_row_shared['ID'].iloc[0] not in human_tasks_df['ID'].values and new_row_shared['ID'].iloc[0] not in robot_tasks_df['ID'].values:
                                robot_tasks_df = pd.concat([robot_tasks_df, new_row_shared], ignore_index = True)
                                if task_df.index[task_df['ID'] == id_shared].tolist()[0] not in indices_to_remove:
                                    #add the index of the shared task to the indices to remove
                                    indices_to_remove.update(shared_row.index.tolist())

                        # add the task to the indices to remove
                        indices_to_remove.add(index)


    else:

        # Iterate over task_df and assign tasks based on tool availability
        for index in reversed(task_df.index):

            # Check if the index is in the indices to remove
            if index in indices_to_remove:

                continue

            else:
                tools_in_task = set(task_df.at[index, 'END-EFFECTOR'].split(', '))

                # Check if available tools intersect with needed tools

                if mounted_tool not in tools_in_task:
                    # Create a temporary DataFrame with the task to add to human_tasks_df
                    task_to_add = pd.DataFrame({
                        'ID': [int(task_df.at[index, 'ID'])],
                        'TASK': [task_df.at[index, 'TASK']],
                        'OBJECT': [task_df.at[index, 'OBJECT']],
                        'PRECEDENT': [task_df.at[index, 'PRECEDENT']],
                        'SHARED': [task_df.at[index, 'SHARED']]
                    })

                    # Add the task to the Human Tasks DataFrame
                    human_tasks_df = pd.concat([human_tasks_df, task_to_add], ignore_index=True)

                    if task_df.at[index, 'SHARED'] != 0:

                        id_shared = task_df.at[index, 'SHARED'].astype(int)
                        shared_row = task_df.loc[task_df['ID'] == id_shared]
                        new_row_shared = pd.DataFrame({
                            'ID': [shared_row['ID'].iloc[0]],
                            'TASK': [shared_row['TASK'].iloc[0]],
                            'OBJECT': [shared_row['OBJECT'].iloc[0]],
                            'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]],
                            'END-EFFECTOR': [shared_row['END-EFFECTOR'].iloc[0]],
                            'SHARED': [shared_row['SHARED'].iloc[0]]
                        })
                        robot_tasks_df = pd.concat([robot_tasks_df, new_row_shared], ignore_index = True)
                        # check if the shared task id is in the indices to remove
                        if task_df.index[task_df['ID'] == id_shared].tolist()[0] not in indices_to_remove:
                            indices_to_remove.update(shared_row.index.tolist())

                    # Add the task to the indices to remove
                    indices_to_remove.add(index)

    # Remove the tasks from the task_df DataFrame
    task_df.drop(list(indices_to_remove), inplace=True)
    # Reset indexes after changes
    task_df.reset_index(drop=True, inplace=True)
    human_tasks_df.reset_index(drop=True, inplace=True)
    return task_df, human_tasks_df, robot_tasks_df




def Layer_3(task_df, assembly_df, distance_matrix_df, human_tasks_df, robot_tasks_df, AUTOMATABILITY_THRESHOLD):

    #COMPONENT 

    # SYMMETRY
    #Alpha (α) refers to the rotational symmetry of a part around an axis perpendicular to its axis of insertion. For example, if a part can be rotated around this axis and repeats its orientation every 180°, it has an α symmetry of 180°.
    #Beta (β) measures the rotational symmetry of a part about its axis of insertion. For instance, a cylinder that can be inserted into a round hole without requiring rotation about its axis of insertion would have a β symmetry of 0°.
    #Based on α and β symmetries, here are the detailed scoring guidelines for grip ease:
    #0% (Very Difficult to Grip): Parts where both α and β are far from the ideal symmetry values (e.g., both are significantly different from 180° for α and 0° for β), requiring complex grippers with multi-directional manipulation or multiple grippers for stable handling.
    #25% (Difficult to Grip): Parts with moderate symmetry where α or β is somewhat symmetric (e.g., one is close to 180°, and the other is at an intermediate value), which can be handled with standard grippers but require careful orientation for correct alignment.
    #50% (Moderately Easy to Grip): Parts with reasonable symmetry (α or β near 180°), allowing the use of standard grippers without the need for precise orientation.
    #75% (Easy to Grip): Parts with good symmetry on at least one axis (either α or β is low, such as one being 0° or 180°), which can be easily handled by a standard gripper without additional orientation.
    #100% (Very Easy to Grip): Parts with perfect symmetry (both α and β are low, ideally α is 180° and β is 0°), which can be grasped by a simple gripper in any orientation without alignment issues.

    # SENSITIVITY
    #0% Highly sensitive
    #25% Damage in careless handling 
    #50% Damage in light force
    #75% Damage at high force
    #100% Robust

    # DIMENSIONAL STABILITY
    #0% Shapeless 
    #25% Deformation possible 
    #50% Deformation under force
    #75% Deformation at high force
    #100% Rigid

    #MOUNTING

    # INSERTION DIRECTION:
    # Vertical from above (with gravitational support): 100%
    #Horizontal (without gravitational support): 50%
    #non-perpendicular direction (challenging for orientation and placement): 0%

    #MOUNTING_RESISTANCE:
    #0% No force required
    #50% Light force required
    #100% Must be pressed with force

    #MOUNTING TOLERANCE:
    #0% < 0,5mm 
    #50% ≥ 0,5 < 1 mm 
    #100% > 1 mm

    #FEEDING

    # PICK-PLACE DISTANCE
    #0% >100 cm 
    #25% 80-100cm
    #50% 60-80 cm 
    #75% 30-60 cm 
    #100% 0-30 cm

    #FEEDING_PRESENTATION 
    #100% Known position and pose of components
    #75% Organized but pose identification required
    #25% Disorganized in a box or on plane
    #0% Components vary widely, needing custom solutions for orientation and handling

    # SAFETY

    #SAFETY_TOOLS [0-2-3-4] (tools increasing safety risks (sharp edged and pointed tools))
    #0% The use would increase the danger to humans
    #50% No consequences
    #75% Risk would be eliminated with safety devices
    #100% The use would reduce danger to humans

    #SAFETY_COLLISION [0-1-2-3-4] (risk of collision with the robot)
    #0% Very high risk of collision
    #25% Frequent risk of collision
    #50% Low risk of collision
    #75% Risk would be eliminated with safety devices
    #100% No risk of collision

    #FASTENING

    #25% Riveting 
    #50% Bending
    #75% Gluing or Screwing

    # Create two new columns in 'task_df' to store the CP and Automatability Potential values

    task_df['CP'] = 0.0
    task_df['AUTOMATABILITY'] = 0.0
    indices_to_remove = []

    # Evaluate the CP and AUTOMATABILITY potential for each task that was not assigned to the human

    for index, task in task_df.iterrows():
        
        # Save the object name
        
        obj = task['OBJECT']
        component = assembly_df[assembly_df['OBJECT'] == obj].iloc[0]

        # Evaluate the CP index based on the physical characteristics of the component
        
        weight = component['WEIGHT (g)']
        weight_score = ut.calculate_weight_score(weight)
        cp_score = (component['SENSITIVITY (0-4)'] + component['DIMENSIONAL STABILITY (0-4)'] +
                    component['SYMMETRY (0-4)'] + weight_score) / 4

        # 'Safety' category
    
        safety_score = (task['SAFETY_TOOLS [0-2-3-4]'] + task['SAFETY_COLLISION (0-4)']) / 2
        
        total_score = safety_score + cp_score
        # Initialize the categories count to 2 because of the CP index and the Safety index (always present)
        categories_count = 2

        # 'Mounting' category
        
        if task.get('MOUNTING_APPLICABLE', False):
            
            mounting_score = (task['MOUNTING_TOLERANCE [0-2-4]'] + task['MOUNTING_DIRECTION [0-2-4]']+ task['MOUNTING_RESISTANCE [0-2-4]']) / 3
            total_score += mounting_score
            categories_count += 1

        # 'Feeding' category
        
        if task.get('FEEDING_APPLICABLE', False):
            distance = distance_matrix_df[distance_matrix_df['OBJECT'] == obj]['ROBOT_DISTANCE_PICK(mm)'].values[0]
            distance_score = ut.calculate_distance_score(distance)
            feeding_score = (distance_score + task['FEEDING_PRESENTATION [0-1-3-4]']) / 2
            total_score += feeding_score
            categories_count += 1


        # 'Fastening' category
        
        if task.get('FASTENING_APPLICABLE', False):
            
            fastening_score = task['FASTENING [1-2-3]']
            total_score += fastening_score
            categories_count += 1


        # Evaluate the final AUTOMATABILITY potential
        
        automatability_score = round(total_score / categories_count,2)

        # Assign the scores to the dataframe
        
        task_df.at[index, 'CP'] = cp_score
        task_df.at[index, 'AUTOMATABILITY'] = automatability_score
       

    for index in reversed(task_df.index):
    
        # Check if the AUTOMATABILITY potential is below 'AUTOMATABILITY_THRESHOLD'

        
        if task_df.at[index, 'ID'] not in human_tasks_df['ID'] and task_df.at[index, 'ID'] not in robot_tasks_df['ID']:

                
            if task_df.at[index, 'AUTOMATABILITY'] < AUTOMATABILITY_THRESHOLD:
                
                # Create a temporary DataFrame with the task to be added
                
                task_to_add = pd.DataFrame({
                    'ID': [int(task_df.at[index, 'ID'])],
                    'TASK': [task_df.at[index, 'TASK']],
                    'OBJECT': [task_df.at[index, 'OBJECT']],
                    'PRECEDENT': [task_df.at[index, 'PRECEDENT']],
                    'SHARED': [task_df.at[index, 'SHARED']]
                })

                # Add the task to the matrix 'human_tasks_df'
                human_tasks_df = pd.concat([human_tasks_df, task_to_add], ignore_index=True)

                if task_df.at[index, 'SHARED'] != 0:

                    id_shared = int(task_df.at[index, 'SHARED'])
                    shared_row = task_df.loc[task_df['ID'] == id_shared]
                    if (not shared_row.empty):
                        new_row_shared = pd.DataFrame({
                            'ID': [shared_row['ID'].iloc[0]],
                            'TASK': [shared_row['TASK'].iloc[0]],
                            'OBJECT': [shared_row['OBJECT'].iloc[0]],
                            'PRECEDENT': [shared_row['PRECEDENT'].iloc[0]],
                            'END-EFFECTOR': [shared_row['END-EFFECTOR'].iloc[0]],
                            'SHARED': [shared_row['SHARED'].iloc[0]]
                        })
                        if (new_row_shared['ID'].values[0] not in human_tasks_df['ID'].values) and (new_row_shared['ID'].values[0] not in robot_tasks_df['ID'].values):
                            robot_tasks_df = pd.concat([robot_tasks_df, new_row_shared], ignore_index = True)
                            indices_to_remove.append(shared_row.index.tolist())

                indices_to_remove.append(index)
    
    # Reset the indices after the modifications

    # Modifica la comprensione di lista per gestire correttamente singoli interi e liste
    flat_indices_to_remove = [item for sublist in indices_to_remove for item in (sublist if isinstance(sublist, list) else [sublist])]


    # Usare la lista appiattita per rimuovere le righe
    task_df.drop(flat_indices_to_remove, inplace=True)

    task_df.reset_index(drop = True, inplace = True)
    human_tasks_df.reset_index(drop = True, inplace = True)

    # Substitute 'NaN' with 0 in 'human_tasks_df' for columns if necessary

    human_tasks_df.fillna(0, inplace=True)
    task_df.fillna(0, inplace=True)

    return task_df, human_tasks_df, robot_tasks_df

def Layer_5(task_df, end_table, position_matrix, human_tasks_df, robot_tasks_df):

    # With this choice, we consider the possible range the part of the table between the robot and the end of the table

    range = end_table
    
    indices_to_remove = []

    for index, row in task_df.iterrows():

        if index not in indices_to_remove:

            # Extract the y coordinates of pick and place
            pick_coord = position_matrix.loc[position_matrix['TASK'] == row['ID'], 'PICK_(X,Y,Z)'].values[0]
            place_coord = position_matrix.loc[position_matrix['TASK'] == row['ID'], 'PLACE_(X,Y,Z)'].values[0]
            
            y_pick = pick_coord[1]
            y_place = place_coord[1]
            
            position_robot = max(y_pick, y_place)
            position_human = min(y_pick, y_place)

            risk_robot = abs(position_robot / range)
            risk_human = abs(position_human / range)

            print(risk_robot, risk_human)

            #the gain will be the time of the robot to complete that task minus the time of the human to complete that task
            #for the time of the robot we'll consider the smallest time with the tool mounted

            time_robot = min([row['tr_gripper_mounted'], row['tr_vacuum_mounted'], row['tr_magnetic_mounted']])

            gain = (time_robot - row['th']) / min(time_robot, row['th'])

            filtered_row_human = row[human_tasks_df.columns.intersection(row.index)]
            filtered_row_robot = row[robot_tasks_df.columns.intersection(row.index)]
            
            if risk_robot > 0.9:
                #assegnamo il task all'umano
                task_df.at[index, 'ASSIGNED'] = 'Human'
                human_tasks_df = pd.concat([human_tasks_df, filtered_row_human.to_frame().T])
                indices_to_remove.append(index)

            elif risk_human < 0.1:
                #assegnamo il task al robot
                task_df.at[index, 'ASSIGNED'] = 'Robot'
                robot_tasks_df = pd.concat([robot_tasks_df, filtered_row_robot.to_frame().T])
                indices_to_remove.append(index)

            if risk_robot > 0.4 and gain > 0:
                #assegnamo il task all'umano
                task_df.at[index, 'ASSIGNED'] = 'Human'
                human_tasks_df = pd.concat([human_tasks_df, filtered_row_human.to_frame().T])
                indices_to_remove.append(index)

            if risk_robot > 0.5 and risk_robot < 0.9 and gain < 0:
                risk_gain = risk_robot - gain
                if risk_gain > 0:
                    #assegnamo il task all'umano
                    task_df.at[index, 'ASSIGNED'] = 'Human'
                    human_tasks_df = pd.concat([human_tasks_df, filtered_row_human.to_frame().T])
                    indices_to_remove.append(index)

            #we check that the task (if added to the index) is also shared
            #if the task is shared and present in indices to remove, then we also add the shared task

            if row['SHARED'] != 0:
                id_shared = row['SHARED']
                shared_row = task_df.loc[task_df['ID'] == id_shared]
                if shared_row.index[0] in indices_to_remove:
                    indices_to_remove.append(shared_row.index[0])
                    if shared_row['ASSIGNED'].values[0] == 'Human':
                        filtered_row_human = shared_row[human_tasks_df.columns.intersection(shared_row.index)]
                        human_tasks_df = pd.concat([human_tasks_df, filtered_row_human.to_frame().T])
                    else:
                        filtered_row_robot = shared_row[robot_tasks_df.columns.intersection(shared_row.index)]
                        robot_tasks_df = pd.concat([robot_tasks_df, filtered_row_robot.to_frame().T])

    #let's remove the lines from the task_df
    task_df.drop(indices_to_remove, inplace=True)

    task_df.reset_index(drop = True, inplace = True)

    return task_df, human_tasks_df, robot_tasks_df


