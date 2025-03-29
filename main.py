import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram

# Step 1: Define the Bank Layout Graph with weighted edges
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

# Add edges with weights (format: (node1, node2, weight))
weighted_edges = [
    (name_to_index["entrance"], name_to_index["Lobby"], 1.0),
    (name_to_index["entrance"], name_to_index["front desk"],2.0),
    (name_to_index["Lobby"], name_to_index["front desk"], 2.0),
    (name_to_index["Lobby"], name_to_index["control room"], 1.0),
    (name_to_index["Lobby"], name_to_index["Office"], 1.0),
    (name_to_index["front desk"], name_to_index["control room"], 1.0),
    (name_to_index["control room"], name_to_index["Office"], 2.0),
    (name_to_index["Security"], name_to_index["Office"], 2.0),
    (name_to_index["Security"], name_to_index["Laser beam"], 3.0),
    (name_to_index["Laser beam"], name_to_index["Vault 1"], 1.0),
    (name_to_index["Laser beam"], name_to_index["Vault 2"], 1.0)
]

bank_graph.add_weighted_edges_from(weighted_edges)

# Step 2: Convert Weighted Graph to QUBO using Maxcut
maxcut = Maxcut(bank_graph)
qubo = maxcut.to_quadratic_program()

# Step 3: Convert QUBO to Ising Hamiltonian
ising, ising_offset = qubo.to_ising()

# Step 4: Create QAOA Circuit
qaoa_reps = 2  # Number of QAOA layers
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps, name='qaoa')
qaoa_ansatz.measure_all()

# Step 5: Assign Parameters
np.random.seed(42)  # For reproducibility
betas = np.random.uniform(0, np.pi, qaoa_reps)
gammas = np.random.uniform(0, 2 * np.pi, qaoa_reps)
parameter_values = np.concatenate((betas, gammas))
param_dict = dict(zip(qaoa_ansatz.parameters, parameter_values))
qaoa_with_parameters = qaoa_ansatz.assign_parameters(param_dict)

# Step 6: Execute on Simulator
aer_simulator = AerSimulator()
compiled_qaoa = transpile(qaoa_with_parameters, aer_simulator)
result = aer_simulator.run(compiled_qaoa, shots=10000).result()  # Reduced shots for faster execution
counts = result.get_counts()

# Step 7: Filter out trivial solutions
def is_trivial_solution(binary_string):
    return binary_string == '0' * len(binary_string) or binary_string == '1' * len(binary_string)

filtered_counts = {k: v for k, v in counts.items() if not is_trivial_solution(k)}

# Step 8: Calculate Weighted Max-Cut classically
def calculate_weighted_max_cut(graph):
    max_weight = 0
    best_partition = None
    num_nodes = len(graph.nodes)
    
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node))]
        set_b = [node for node in range(num_nodes) if not (i & (1 << node))]
        
        cut_weight = sum(graph[u][v]['weight'] 
                     for u, v in graph.edges 
                     if (u in set_a and v in set_b) or (u in set_b and v in set_a))
        
        if cut_weight > max_weight:
            max_weight = cut_weight
            best_partition = (set_a, set_b, [(u,v) for u,v in graph.edges 
                                          if (u in set_a and v in set_b) or (u in set_b and v in set_a)])
    
    return max_weight, best_partition

max_weight, best_partition = calculate_weighted_max_cut(bank_graph)

# Step 9: Print and visualize results
print("\nBank Security Layout Analysis with Weighted Edges")
print("==============================================")
print("Nodes:", node_names)
print("Edges with weights:")
for u, v, w in weighted_edges:
    print(f"  {index_to_name[u]} -- {index_to_name[v]}: weight {w}")

print(f"\nClassical Maximum Weighted Cut: {max_weight}")
print(f"Optimal Partition:")
print(f"  Set A: {[index_to_name[i] for i in best_partition[0]]}")
print(f"  Set B: {[index_to_name[i] for i in best_partition[1]]}")
print("Cut Edges with weights:")
for u, v in best_partition[2]:
    print(f"  {index_to_name[u]} -- {index_to_name[v]}: weight {bank_graph[u][v]['weight']}")

print("\nQAOA Results (Top 5 non-trivial solutions):")
sorted_solutions = sorted(filtered_counts.items(), key=lambda x: -x[1])[:5]
for solution, count in sorted_solutions:
    set_a = [index_to_name[i] for i, bit in enumerate(reversed(solution)) if bit == '1']
    set_b = [name for name in node_names if name not in set_a]
    cut_weight = sum(bank_graph[u][v]['weight'] 
                 for u, v in bank_graph.edges 
                 if (name_to_index[set_a[0]] in [u,v] and name_to_index[set_b[0]] in [u,v]))
    print(f"\nSolution: {solution} (Count: {count}, Estimated Cut Weight: {cut_weight:.1f})")
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")

# Enhanced visualization showing edge weights
plt.figure(figsize=(14, 8))
pos = nx.spring_layout(bank_graph, weight='weight', seed=42)  # Consistent layout

# Draw nodes
nx.draw_networkx_nodes(bank_graph, pos, node_color='lightblue', node_size=2500)

# Draw labels
nx.draw_networkx_labels(bank_graph, pos, labels=index_to_name, font_size=10)

# Draw all edges with width proportional to weight
all_edges = list(bank_graph.edges())
edge_widths = [bank_graph[u][v]['weight']*0.8 for u,v in all_edges]
nx.draw_networkx_edges(bank_graph, pos, edgelist=all_edges, 
                      edge_color='lightgray', width=edge_widths, alpha=0.7)

# Highlight cut edges
cut_edges = best_partition[2]
cut_widths = [bank_graph[u][v]['weight']*1.5 for u,v in cut_edges]
nx.draw_networkx_edges(bank_graph, pos, edgelist=cut_edges, 
                      edge_color='red', width=cut_widths)

# Add edge weight labels
edge_labels = {(u, v): f"{bank_graph[u][v]['weight']:.1f}" for u, v in bank_graph.edges()}
nx.draw_networkx_edge_labels(bank_graph, pos, edge_labels=edge_labels, font_size=9)

plt.title("Bank Security Layout with Weighted Edges\n(Optimal Cut in Red, Width = Weight)", pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# Plot histogram of QAOA results
if filtered_counts:
    plot_histogram(filtered_counts, figsize=(10, 5))
    plt.title("QAOA Solution Distribution (Non-trivial Solutions)")
    plt.show()