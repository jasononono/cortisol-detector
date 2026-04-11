import numpy as np


class Layer:
    trainable = False
    id = 0

    def __init__(self):
        self.parent = None
        self.pass_back = False

    def forward(self, activation, batch_size):
        return activation

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        return gradient


class Convolution(Layer):
    trainable = True
    id = 1

    def __init__(self, input_size, filter_size, input_channels, output_channels, stride):
        super().__init__()

        self.input_size = input_size # i
        self.filter_size = filter_size # f
        self.input_channels = input_channels # I
        self.output_channels = output_channels # O
        self.stride = stride
        self.output_size = (input_size - filter_size) // stride + 1 # o

        self.filters = None # (I, O, f, f)
        self.filters_col = None # (O, If^2)
        self.channel_biases = None # (O)
        self.biases_col = None # (O, 1)
        self.kernels = None # (B, If^2, o^2)

    def init_parameter(self):
        self.filters = np.random.randn(self.input_channels, self.output_channels, self.filter_size, self.filter_size) * np.sqrt(2 / (self.filter_size ** 2 * self.input_channels))
        self.filters_col = self.filters.transpose(1, 0, 2, 3).reshape(self.output_channels, self.input_channels * self.filter_size ** 2)
        self.channel_biases = np.zeros(self.output_channels)
        self.biases_col = self.channel_biases.reshape(self.output_channels, 1)

    def set_parameter(self, f, b):
        self.filters = f
        self.filters_col = self.filters.transpose(1, 0, 2, 3).reshape(self.output_channels, self.input_channels * self.filter_size ** 2)
        self.channel_biases = b
        self.biases_col = self.channel_biases.reshape(self.output_channels, 1)

    def im2col(self, activation, batch_size):
        kernels = np.lib.stride_tricks.as_strided(
            activation,
            shape = (batch_size, self.input_channels,
                     self.output_size, self.output_size,
                     self.filter_size, self.filter_size),
            strides = (activation.strides[0], activation.strides[1],
                       activation.strides[2] * self.stride, activation.strides[3] * self.stride,
                       activation.strides[2], activation.strides[3])
        ) # (B, I, o, o, f, f)
        return kernels.transpose(0, 1, 4, 5, 2, 3).reshape(batch_size, self.input_channels * self.filter_size ** 2, self.output_size ** 2)

    def col2im(self, gradient, batch_size):
        delta = gradient @ self.filters_col # (B, o^2, If^2)
        delta = delta.reshape(batch_size, self.output_size, self.output_size, self.input_channels, self.filter_size, self.filter_size).transpose(0, 3, 1, 2, 4, 5) # (B, I, o, o, f, f)

        gradient = np.zeros((batch_size, self.input_channels, self.input_size, self.input_size))
        view = np.lib.stride_tricks.as_strided(
            gradient,
            shape = (batch_size, self.input_channels,
                     self.output_size, self.output_size,
                     self.filter_size, self.filter_size),
            strides = (gradient.strides[0], gradient.strides[1],
                       gradient.strides[2] * self.stride, gradient.strides[3] * self.stride,
                       gradient.strides[2], gradient.strides[3])
        ) # (B, I, o, o, f, f)

        for i in range(self.output_size):
            for j in range(self.output_size):
                view[:, :, i, j] += delta[:, :, i, j]
        # equivalently np.add.at(view, slice(None), delta)

        return gradient

    def forward(self, activation, batch_size):
        self.kernels = self.im2col(activation, batch_size) # (B, If^2, o^2)
        activation = self.filters_col @ self.kernels + self.biases_col # (B, O, o^2)
        return activation.reshape(batch_size, self.output_channels, self.output_size, self.output_size)

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        gradient = gradient.transpose(0, 2, 3, 1).reshape(batch_size, self.output_size ** 2, self.output_channels) # (B, o^2, O)
        delta_filter = (self.kernels @ gradient).transpose(0, 2, 1) # (B, O, If^2)
        delta_biases = np.sum(gradient, axis = (0, 1)) # (O)
        if self.pass_back:
            gradient = self.col2im(gradient, batch_size)
        self.filters_col -= np.mean(delta_filter, 0) * delta_multiplier
        self.channel_biases -= delta_biases * delta_multiplier
        return gradient


class ReLU(Layer):
    id = 2

    def __init__(self):
        super().__init__()
        self.activation_prev = None

    def forward(self, activation, batch_size):
        self.activation_prev = activation
        return np.maximum(activation, 0)

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        if self.pass_back:
            gradient = gradient * np.where(self.activation_prev > 0, 1, 0)
        return gradient


class Softmax(Layer):
    id = 3

    def forward(self, activation, batch_size):
        e = np.exp(activation - np.max(activation, axis = -1, keepdims = True))
        return e / np.sum(e, axis = -1, keepdims = True)


class Reshape(Layer):
    id = 4

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
        self.output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

    def forward(self, activation, batch_size):
        return activation.reshape(-1, *self.output_shape)

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        if self.pass_back:
            gradient =  gradient.reshape(-1, *self.input_shape)
        return gradient


class Dense(Layer):
    trainable = True
    id = 5

    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.weights = None
        self.biases = None
        self.activation = None
        self.activation_prev = None

    def init_parameter(self):
        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        self.biases = np.zeros(self.output_size)

    def set_parameter(self, w, b):
        self.weights = w
        self.biases = b

    def forward(self, activation, batch_size):
        self.activation_prev = activation
        self.activation = activation @ self.weights + self.biases
        return self.activation

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        delta_weights = self.activation_prev.T @ gradient * delta_multiplier
        delta_biases = np.sum(gradient, axis = 0) * delta_multiplier
        if self.pass_back:
            gradient = gradient @ self.weights.T
        self.weights -= delta_weights
        self.biases -= delta_biases
        return gradient


class MaxPool(Layer):
    id = 6

    def __init__(self, input_size, channels, filter_size):
        super().__init__()

        self.input_size = input_size
        self.channels = channels
        self.filter_size = filter_size
        self.output_size = input_size // filter_size

        self.kernels = None

    def forward(self, activation, batch_size):
        kernels = np.lib.stride_tricks.as_strided(
            activation,
            shape = (batch_size, self.channels,
                     self.output_size, self.output_size,
                     self.filter_size, self.filter_size),
            strides = (activation.strides[0], activation.strides[1],
                       activation.strides[2] * self.filter_size, activation.strides[3] * self.filter_size,
                       activation.strides[2], activation.strides[3])
        )
        kernels = kernels.reshape(batch_size, self.channels, self.output_size, self.output_size, self.filter_size ** 2)
        self.kernels = np.argmax(kernels, axis = 4)
        return np.take_along_axis(kernels, self.kernels[..., None], axis = -1).reshape(batch_size ,self.channels, self.output_size, self.output_size)

    def backpropagate(self, gradient, delta_multiplier, batch_size):
        delta = np.zeros((batch_size, self.channels, self.input_size ** 2))

        for i in range(self.output_size):
            for j in range(self.output_size):
                indices = (i * self.filter_size + self.kernels[:, :, i, j] // self.filter_size) * self.input_size + (j * self.filter_size + self.kernels[:, :, i, j] % self.filter_size)
                np.put_along_axis(delta, indices[..., None], gradient[:, :, i, j, None], axis = -1)

        return delta.reshape(batch_size, self.channels, self.input_size, self.input_size)