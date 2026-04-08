import numpy as np


class Layer:
    trainable = False
    id = 0

    def forward(self, activation):
        return activation

    def backpropagate(self, gradient, delta_multiplier):
        return gradient


class Convolution(Layer):
    trainable = True
    id = 1

    def __init__(self, input_size, filter_size, stride = 1):
        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = (input_size - filter_size) // stride + 1
        self.stride = stride

        self.filter = None
        self.filter_flat = None
        self.kernels = None

    def init_parameter(self):
        self.filter = np.random.randn(self.filter_size, self.filter_size)
        self.filter_flat = self.filter.reshape(self.filter_size ** 2)

    def set_parameter(self, f):
        self.filter = f
        self.filter_flat = self.filter.reshape(self.filter_size ** 2)

    def im2col(self, activation):
        kernels = np.lib.stride_tricks.as_strided(
            activation,
            shape = (activation.shape[0],
                     self.output_size, self.output_size,
                     self.filter_size, self.filter_size),
            strides = (activation.strides[0],
                       activation.strides[1] * self.stride, activation.strides[2] * self.stride,
                       activation.strides[1], activation.strides[2])
        )
        return kernels.reshape(activation.shape[0], self.output_size ** 2, self.filter_size ** 2)

    def col2im(self, gradient):
        delta = gradient * self.filter_flat
        delta = delta.reshape(delta.shape[0], self.output_size, self.output_size, self.filter_size, self.filter_size)

        gradient = np.zeros((self.kernels.shape[0], self.input_size, self.input_size))
        view = np.lib.stride_tricks.as_strided(
            gradient,
            shape = (gradient.shape[0],
                     self.output_size, self.output_size,
                     self.filter_size, self.filter_size),
            strides = (gradient.strides[0],
                       gradient.strides[1] * self.stride, gradient.strides[2] * self.stride,
                       gradient.strides[1], gradient.strides[2])
        )

        for i in range(self.output_size):
            for j in range(self.output_size):
                view[:, i, j] += delta[:, i, j]

        return gradient

    def forward(self, activation):
        self.kernels = self.im2col(activation)
        activation = self.kernels @ self.filter_flat
        return activation.reshape(activation.shape[0], self.output_size, self.output_size)

    def backpropagate(self, gradient, delta_multiplier):
        gradient = gradient.reshape(gradient.shape[0], self.output_size ** 2, 1)
        delta = self.kernels.transpose(0, 2, 1) @ gradient
        gradient = self.col2im(gradient)
        self.filter_flat -= np.mean(delta, 0).reshape(self.filter_size ** 2) * delta_multiplier
        return gradient


class ReLU(Layer):
    id = 2

    def __init__(self):
        self.activation_prev = None

    def forward(self, activation):
        self.activation_prev = activation
        return np.maximum(activation, 0)

    def backpropagate(self, gradient, delta_multiplier):
        return gradient * np.where(self.activation_prev > 0, 1, 0)


class Softmax(Layer):
    id = 3

    def forward(self, activation):
        e = np.exp(activation - np.max(activation, axis = -1, keepdims = True))
        return e / np.sum(e, axis = -1, keepdims = True)


class Reshape(Layer):
    id = 4

    def __init__(self, input_shape, output_shape):
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
        self.output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

    def forward(self, activation):
        return activation.reshape(-1, *self.output_shape)

    def backpropagate(self, gradient, delta_multiplier):
        return gradient.reshape(-1, *self.input_shape)


class Dense(Layer):
    trainable = True
    id = 5

    def __init__(self, input_size, output_size):
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

    def forward(self, activation):
        self.activation_prev = activation
        self.activation = activation @ self.weights + self.biases
        return self.activation

    def backpropagate(self, gradient, delta_multiplier):
        delta_weights = self.activation_prev.T @ gradient * delta_multiplier
        delta_biases = np.sum(gradient, axis = 0) * delta_multiplier
        gradient = gradient @ self.weights.T
        self.weights -= delta_weights
        self.biases -= delta_biases
        return gradient