

""" 
Definition of some initial parameters
* verbose = True if you want to print the tables
* precedence_graph_plot = True if you want to display the precedence graph
* update_matrix = True if you want to update the time matrix at the end
* n = number of objects
* robot_max_load = maximum payload for the robot (in grams [g])
* robot_max_distance = maximum distance with respect to the robot (expressed in millimeters [mm])
* pos_lb, pos_ub = lower and upper bounds for the random generation of the positions of the objects (in millimeters [mm])
* w_lb, w_ub = lower and upper bounds for the random generation of the weights of the objects (in grams [g])
* z_obj = height of the objects (in millimeters [mm])
* human_position = 'fictitious' position of the human (in millimeters [mm])
* robot_position = 'fictitious' position of the robot (in millimeters [mm])
* HRC_THRESHOLD = threshold for the HRC potential (initial choice to be evaluated)
* division_constraint = constraint on the minimum division of tasks
* mounted_tool = tool currently mounted on the robot
* available_tools = list of tools available for the robot (the mounted tool and the tools in the turret)
* RULA_THRESHOLD = threshold for the RULA score (initial choice to be evaluated)
* RULA_AVERAGE_THRESHOLD = threshold for the average RULA score (initial choice to be evaluated)
* rula_tot = sum of the RULA scores of the tasks assigned to the human
* rula_average = average RULA score of the tasks assigned to the human
* tool_change_feasible = True if the tool change is feasible, False otherwise
* tool_change_time = time required to change the tool (in seconds)
* end_table = end of the table to the side of the human in mm
"""

verbose = True
ifrandom = True
precedence_graph_plot = True
update_matrix = False
n = 12
robot_max_load = 5000
robot_max_distance = 850
#table_x_dimension = 950 mm
pos_x_lb = -400
pos_x_ub = 400
#table_y_dimension = 1175 mm
pos_y_lb = 0
pos_y_ub = 1100
w_lb = 100
w_ub = 6000
z_obj = 0
human_position = (1050, 0, 0)
robot_position = (0, 0, 0)
AUTOMATABILITY_THRESHOLD = 30
division_constraint = 0.25
mounted_tool = 'gripper'
available_tools = ['gripper', 'vacuum', 'magnetic']
RULA_THRESHOLD = 5
RULA_AVERAGE_THRESHOLD = 4
rula_tot = 0
tool_change_feasible = True
tool_change_time = 6.0
end_table = 1000
