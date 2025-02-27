#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #

    cost_h, mixer_h = qml.qaoa.cost.max_independent_set(graph)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        qml.layer(qaoa_layer, N_LAYERS, params[0], params[1])

    dev = qml.device("default.qubit", wires=NODES)

##    @qml.qnode(dev)
##    def cost_function(params):
##        circuit(params)
##        return qml.expval(cost_h)
##
##    optimizer = qml.GradientDescentOptimizer()
##    for i in range(70):
##        params = optimizer.step(cost_function, params)

    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=range(NODES))

    probs = probability_circuit(params[0], params[1])

    max_ind_set = []

##    binary_best = ("{:0" + str(NODES) + "b}").format(probs.argmax())
##    for i in range(NODES):
##        if binary_best[i] == '1':
##            max_ind_set.append(i)

    best = probs.argmax()
    for i in range(NODES-1,-1,-1):
        if best % 2 == 1:
            max_ind_set.append(i)
        best >>= 1
    max_ind_set.reverse()

    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
