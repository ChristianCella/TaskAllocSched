# Layout – driven task allocation and scheduling​
 
## Related documentation
- 

## Overview

This repository contains a Python-based implementation designed to optimize task allocation and scheduling in a human-robot collaborative environment. The project focuses on efficiently distributing tasks between humans and robots, minimizing time, ergonomic risks, and ensuring task distribution adheres to established limits, with no less than 30% and no more than 70% of tasks allocated to either humans or robots.  In this project a logic model has been implemented to effectively manage the robot's tool changes. 

## Project Structure

### Layer 0: Matrix Generation

- **Layer 0.1 and Layer 0.2: Components**: Generate a realistic set of tasks and objects in an assembly setting using randomized parameters (in layer 0.1 the focus is on quantities related to the objects, like positions and weights, while in layer 0.2 the focus is on the precedence, end effector, tasks, ...).
- **Layer 0.3: Distance Matrix**: First of all, the matrix (composed of 2 columns) for the pick and the place positions is generated; Secondly, a new table that conatins the distances between each object and both human and robot is calculated, in order to facilitate task assignment based on proximity.
- **Layer 0.4: Precedenche graph**: Establish task dependencies using a directed graph, ensuring tasks are executed in a logical order.
- **Layer 0.5: 'compulsory tasks'**: Check if there are some tasks that must be specifically assigned to an agent and exclude them for future passages.

### Layer 1: Maximum Load and Distance Evaluation

- **Task Assignment**: Automatically assign tasks exceeding robot capabilities in weight and distance to humans.

### Layer 2: Tool Evaluation

- **Tool Suitability**: Assign tasks to humans when the robot's available tools are not suitable for the task.

### Layer 3: HRC Potential Evaluation

- **Automation Suitability**: Allocate tasks to humans that are less suitable for automation, based on a complexity-based algorithm, through the calculation of HRC potential.

### Pre-Simulation

- **Simulation Preparation**: Prepare tasks with realistic timings and RULA scores, obtained from Tecnomatrix Process Simulate simulations.

### Layer 4: RULA Evaluation

- **Ergonomic Assignment**: Assign tasks likely to induce high RULA scores to robots to mitigate ergonomic risk to human workers.

### Task Allocation

- **Optimization Algorithm**: Implement a Branch-and-Bound algorithm to find the best task distribution that minimizes total task time and keeps the average RULA score for human tasks below a threshold.

### Scheduling Optimization

- **MILP Implementation**: Use Mixed-Integer Linear Programming to optimize the sequence and timing of tasks, ensuring efficient progression and adherence to dependencies.

### Visualization

- **Gantt Chart**: Visualize task allocation and scheduling, clearly showing the distribution of tasks over time between human and robot, as well as the specific tools each robot task employs.

## Installation and Setup

Ensure you have Python 3.7 or later installed. 
