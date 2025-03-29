import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram


# Step 1: Define the Bank Layout Graph with proper node indexing
node_names = [
    "Lobby", "entrance", "front desk",
    "control room", "Laser beam", "Security", "Office",
    "Vault 1", "Vault 2"
]

# Create mapping between names and indices
name_to_index = {name: i for i, name in enumerate(node_names)}
index_to_name = {i: name for i, name in enumerate(node_names)}

# Initialize graph with integer nodes
bank_graph = nx.Graph()
bank_graph.add_nodes_from(range(len(node_names)))

# Add edges (using integer indices)
edges = [
    (name_to_index["entrance"], name_to_index["Lobby"]),
    (name_to_index["entrance"], name_to_index["front desk"]),
    (name_to_index["Lobby"], name_to_index["front desk"]),
    (name_to_index["Lobby"], name_to_index["control room"]),
    (name_to_index["Lobby"], name_to_index["Office"]),
    (name_to_index["front desk"], name_to_index["control room"]),
    (name_to_index["control room"], name_to_index["Office"]),
    (name_to_index["Security"], name_to_index["Office"]),
    (name_to_index["Security"], name_to_index["Laser beam"]),
    (name_to_index["Laser beam"], name_to_index["Vault 1"]),
    (name_to_index["Laser beam"], name_to_index["Vault 2"])
]
bank_graph.add_edges_from(edges)

# Step 2: Convert Graph to QUBO using Maxcut class
maxcut = Maxcut(bank_graph)
qubo = maxcut.to_quadratic_program()

# Step 3: Convert QUBO to Ising Hamiltonian
ising, ising_offset = qubo.to_ising()

# Step 4: Create QAOA Circuit
qaoa_reps = 2
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps, name='qaoa')
qaoa_ansatz.measure_all()

# Step 5: Assign Parameters
betas = np.random.uniform(0, np.pi, qaoa_reps)
gammas = np.random.uniform(0, 2 * np.pi, qaoa_reps)
parameter_values = [*betas, *gammas]
qaoa_with_parameters = qaoa_ansatz.assign_parameters(dict(zip(qaoa_ansatz.parameters, parameter_values)))

# Step 6: Execute on Simulator
aer_simulator = AerSimulator()
compiled_qaoa = transpile(qaoa_with_parameters, aer_simulator)
result = aer_simulator.run(compiled_qaoa, shots=10**5).result()
counts = result.get_counts()

# Step 7: Discard 0000 and 1111 as these are Not Cuts
def is_trivial_solution(binary_string):
    """
    Check if the solution is trivial (either all 0's or all 1's).
    """
    return binary_string == '0' * len(binary_string) or binary_string == '1' * len(binary_string)

# Step 8: Filter out trivial solutions
filtered_counts = {binary_string: count for binary_string, count in counts.items() if not is_trivial_solution(binary_string)}

# Step 9: Calculate the Max-Cut and Max Number of Cuts
def calculate_max_cut(graph):
    """
    Calculate the maximum number of cuts for a given graph.
    """
    max_cuts = 0
    best_partition = None

    num_nodes = len(graph.nodes)
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]
        
        cut_edges = [(u, v) for u, v in graph.edges if (u in set_a and v in set_b) or (u in set_b and v in set_a)]
        
        if len(cut_edges) > max_cuts:
            max_cuts = len(cut_edges)
            best_partition = (set_a, set_b, cut_edges)

    return max_cuts, best_partition

# Step 10: Calculate the maximum cut for the graph
max_cuts, best_partition = calculate_max_cut(bank_graph)

# Step 11: Output the results
print(f"Maximum number of cuts: {max_cuts}")
print(f"Best partition: Set A: {best_partition[0]}, Set B: {best_partition[1]}")
print(f"Cut edges: {best_partition[2]}")

# Visualize
pos = nx.spring_layout(bank_graph)
nx.draw(bank_graph, pos, with_labels=True, node_color='lightblue')
nx.draw_networkx_edge_labels(bank_graph, pos, edge_labels=nx.get_edge_attributes(bank_graph, 'weight'))
plt.title("Bank Security System")
plt.show()