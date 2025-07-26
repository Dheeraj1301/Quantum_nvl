import pennylane as qml
from pennylane import numpy as np

class VQAOABalancer:
    def __init__(self, wires=4, layers=2):
        self.wires = wires
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=wires)
        self.params = np.random.random((layers, 2))

    def circuit(self, params):
        for i in range(self.wires):
            qml.Hadamard(wires=i)
        for l in range(self.layers):
            for i in range(self.wires):
                qml.RZ(params[l][0], wires=i)
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RY(params[l][1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def optimize(self):
        qnode = qml.QNode(self.circuit, self.dev)
        opt = qml.GradientDescentOptimizer(0.1)
        params = self.params
        for _ in range(30):
            params = opt.step(lambda v: -sum(qnode(v)), params)
        return qnode(params)
