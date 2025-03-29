import matplotlib.pyplot as plt
import networkx as nx 

# Create empty graph
bank = nx.Graph()

# Add nodes (security checkpoints)
bank.add_node(0, name="Entrance", security=1, time_needed=0)
bank.add_node(1, name="Biometric Scanner", security=3, time_needed=2)
bank.add_node(2, name="Laser Grid", security=5, time_needed=3)
bank.add_node(3, name="Vault Door", security=7, time_needed=5)
bank.add_node(4, name="Vault", security=10, time_needed=0)

# Add edges (paths between checkpoints)
bank.add_edge(0, 1, weight=2, description="Main hallway")
bank.add_edge(1, 2, weight=4, description="Laser corridor")
bank.add_edge(2, 3, weight=5, description="Reinforced tunnel")
bank.add_edge(0, 3, weight=6, description="Ventilation shaft (risky!)")
bank.add_edge(3, 4, weight=8, description="Final vault door")

# Visualize
pos = nx.spring_layout(bank)
nx.draw(bank, pos, with_labels=True, node_color='lightblue')
nx.draw_networkx_edge_labels(bank, pos, edge_labels=nx.get_edge_attributes(bank, 'weight'))
plt.title("Bank Security System")
plt.show()


