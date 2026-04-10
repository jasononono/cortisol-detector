import file_util, bin_util, layer


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


def save(model, file_name):
    data = bin_util.HexIO()

    for l in model.layers:
        data.write_uint8(l.id)
        match l.id:
            case 1: # Convolution
                data.write_uint16(l.input_size)
                data.write_uint16(l.filter_size)
                data.write_uint16(l.input_channels)
                data.write_uint16(l.output_channels)
                data.write_uint16(l.stride)
                data.write_numpy(l.filters)
                data.write_numpy(l.channel_biases)
            case 2: # ReLU
                pass
            case 3: # Softmax
                pass
            case 4: # Reshape
                data.write_uint16(len(l.input_shape))
                for i in l.input_shape:
                    data.write_uint16(i)
                data.write_uint16(len(l.output_shape))
                for i in l.output_shape:
                    data.write_uint16(i)
            case 5: # Dense
                data.write_uint16(l.input_size)
                data.write_uint16(l.output_size)
                data.write_numpy(l.weights)
                data.write_numpy(l.biases)
            case 6: # MaxPool
                data.write_uint16(l.input_size)
                data.write_uint16(l.channels)
                data.write_uint16(l.filter_size)
            case _:
                pass

    with open(file_util.path(file_name), "wb") as file:
        file.write(data.data)

def load(file_name):
    with open(file_util.path(file_name), "rb") as file:
        data = bin_util.HexIO(file.read())

    model = Model()

    while data.ptr_valid():
        layer_type = data.read_uint8()
        match layer_type:
            case 1: # Convolution
                l = layer.Convolution(*[data.read_uint16() for _ in range(5)])
                l.set_parameter(data.read_numpy(), data.read_numpy())
            case 2: # ReLU
                l = layer.ReLU()
            case 3: # Softmax
                l = layer.Softmax()
            case 4: # Reshape
                l = layer.Reshape(tuple([data.read_uint16() for _ in range(data.read_uint16())]),
                                    tuple([data.read_uint16() for _ in range(data.read_uint16())]))
            case 5: # Dense
                l = layer.Dense(data.read_uint16(), data.read_uint16())
                l.set_parameter(data.read_numpy(), data.read_numpy())
            case 6: # MaxPool
                l = layer.MaxPool(data.read_uint16(), data.read_uint16(), data.read_uint16())
            case _:
                l = layer.Layer()

        model.add(l)

    return model