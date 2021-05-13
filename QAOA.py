import qiskit as qs

qiskit.__qiskit_version__
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.aqua.operators import op_converter
from qiskit.aqua.operators import WeightedPauliOperator


def hamiltonian_trotter_term(Graph, gamma):
    ''' This produces a Trotter term of the Ising Hamiltonian for Graph'''
    N = Graph.number_of_nodes()
    qc = qs.QuantumCircuit(N, N)
    for j, k in Graph.edges():
        qc = qc.compose(pauli_zz_gate(N, j, k, gamma))
    return qc


def pauli_zz_gate(N, j, k, gamma):
    '''Produces a Pauli $Z_i \otimes Z_j$ gate '''
    qc = qs.QuantumCircuit(N, N)
    qc.cx(j, k)
    qc.rx(2 * gamma, k)
    qc.cx(j, k)
    return qc





def QAOA_Mixer(Graph, beta):
    '''QAOA Mixer Gates rotate each qubit by $\beta$ around x-Axis'''
    N = Graph.number_of_nodes()
    qc = qs.QuantumCircuit(N, N)
    for j in Graph.nodes():
        qc.rx(2 * beta, j)
    return qc





def QAOA_Cirquit(Graph, p, beta, gamma):
    ''' Generator for QAOA circuit from a Graph object, $\beta$ and $\gamma$ arrays '''
    # Ensure correct array lengths
    assert (len(beta) == len(gamma) == p)
    N = Graph.number_of_nodes();
    qc = qs.QuantumCircuit(N, N)
    # Initialize superposition
    qc.h(range(N))
    # Trotter Product of $exp(- \beta B + i \gamma C)$ with p terms
    for k in range(p):
        qc = qc.compose(hamiltonian_trotter_term(Graph, gamma[k]))
        print(k)
        qc = qc.compose(QAOA_Mixer(Graph, beta[k]))
    # barrier and measurement
    qc.barrier(range(N))
    qc.measure(range(N), range(N))
    return qc

def QAOA_Test():
    Graph = nx.Graph()
    Graph.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])
    nx.draw(Graph, pos=nx.bipartite_layout(Graph, [0, 1, 2]))
    qc = QAOA_Mixer(Graph, 1)
    qc.draw('mpl')
    qc = QAOA_Cirquit(Graph, 2, [1, 3], [2, 1])
    qc.draw('mpl')
    qc = pauli_zz_gate(3, 1, 2, np.pi)
    qc.draw('mpl')
    qc = hamiltonian_trotter_term(Graph, np.pi)
    qc.draw('mpl')
