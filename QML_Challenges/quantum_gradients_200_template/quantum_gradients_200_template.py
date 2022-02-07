#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #

    s = 0.1
    n = len(gradient)

    e = []
    for i in range(n):
        basic_vect = np.zeros([5], dtype=np.float64)
        basic_vect[i] = 1
        e.append(basic_vect)

    cache = {}
    def circ_wrap(pert):
        key = pert.tobytes()
        if key not in cache:
            cache[key] = circuit(weights + pert)
        return cache[key]

    def first_deriv(i):
        # Use s = 2s to save an extra computation
        return ( circ_wrap(2*s*e[i]) - circ_wrap(-2*s*e[i]) ) / (2*np.sin(2*s))

    def second_deriv(i,j):
        return (  circ_wrap(s*(e[i] + e[j]))
                - circ_wrap(s*(e[i] - e[j]))
                - circ_wrap(s*(-e[i] + e[j]))
                + circ_wrap(s*(-e[i] - e[j])) ) / (4*np.sin(s)**2)

    for i in range(n):
        gradient[i] = first_deriv(i)
        for j in range(n):
            hessian[i,j] = second_deriv(i,j)

    # QHACK #

    return gradient, hessian, circuit.diff_method


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
