from QAOA_MAX_cut import MAX_Cut, QAOA_Test
import qiskit as qs
import networkx as nx
import matplotlib.pyplot as plt


QAOA_Test()

Graph = nx.erdos_renyi_graph(5, 0.75)
nx.draw(Graph)
plt.show()
backend = qs.Aer.get_backend('qasm_simulator')

cut, solution = MAX_Cut(Graph, backend)
print(" best cut ", cut)
print(" Best Solution", solution)


left = []
for node in Graph:
    if solution[node] == '0':
        left.append(node)

nx.draw(Graph,  pos=nx.bipartite_layout(Graph, left))
plt.show()