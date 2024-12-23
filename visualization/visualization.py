import matplotlib.pyplot as plt
import networkx as nx

def plot_precedence_graph(task_df, G, precedence_graph_plot=True):
    # Assume 'ID' is the unique index and 'TASK' is the name of the task

    # Find tasks that have no predecessors (PRECEDENT == 0)
    tasks_with_no_precedent = [row['TASK'] for index, row in task_df.iterrows() if row['PRECEDENT'] == 0]

    # Add nodes to the graph for each task
    for task in task_df['TASK']:
        G.add_node(task)

    # Add edges based on the 'PRECEDENT' column
    for index, row in task_df.iterrows():
        if row['PRECEDENT'] > 0:  # If there is a precedent task
            # Get the name of the preceding task using its ID
            precedent_task = task_df.loc[task_df['ID'] == row['PRECEDENT'], 'TASK'].values[0]
            # Add an arc from the previous task to this task
            G.add_edge(precedent_task, row['TASK'])

    pos = nx.spring_layout(G, k=1.5)  # Layout for positioning nodes

    if precedence_graph_plot:
        plt.figure(num='Precedence Graph', figsize=(12, 8))
        plt.suptitle('Precedence Graph')

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', linewidths=2, font_size=15, arrows=True)

        # Highlight nodes without predecessors with a black circle
        nx.draw_networkx_nodes(G, pos, nodelist=tasks_with_no_precedent, node_color='skyblue', node_size=3000, edgecolors='black', linewidths=2)

        plt.title("Task Precedence Graph")
        plt.axis('off')  # Hide the axes for a cleaner display

        # Display the graph
        plt.show(block=True)
