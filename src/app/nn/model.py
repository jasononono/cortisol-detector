from . import util, layer


class Model:
    def __init__(self):
        self.layers = []

    def forward(self, activation, batch_size):
        for l in self.layers:
            activation = l.forward(activation, batch_size)
        return activation

    def compile(self):
        for i in self.layers:
            if i.trainable:
                i.init_parameter()

    def add(self, l):
        l.parent = self
        l.pass_back = len(self.layers) != 0
        self.layers.append(l)

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        for i in range(len(self.layers) - 2, -1, -1):
            gradient = self.layers[i].backpropagate(gradient, delta_multiplier, batch_size)