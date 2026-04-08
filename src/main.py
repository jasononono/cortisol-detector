import numpy as np
import cProfile
import model
from layer import Convolution, ReLU, Reshape, Dense, Softmax
import training



data_in = np.load("src/data_mnist/input_train.npy")
data_out = np.load("src/data_mnist/output_train.npy")
data_in = data_in.reshape(-1, 1, 28, 28)


# conv = Convolution(28, 3, 1, 4, 1)
# conv.init_parameter()
#
# print(conv.forward(data_in[0].reshape(1, 1, 28, 28)))


nn = model.Model()
nn.add(Convolution(28, 3, 1, 1, 1))
nn.add(ReLU())
nn.add(Reshape((1, 26, 26), 676))
nn.add(Dense(676, 10))
nn.add(Softmax())


try:
    training.batched(nn, data_in, data_out, 100)
except KeyboardInterrupt:
    file_name = input("save model as (leave blank to discard) > ")
    if file_name:
        model.save(nn, "src/models/" + file_name)


# data_in = np.load("src/data/train_input.npy")
# data_out = np.load("src/data/train_output.npy")
# valid_in = np.load("src/data/test_input.npy")
# valid_out = np.load("src/data/test_output.npy")
#
#
# nn = Model()
# nn.add(Convolution(48, 5, 1))
# nn.add(ReLU())
# nn.add(Reshape((44, 44), 1936))
# nn.add(Dense(1936, 32))
# nn.add(ReLU())
# nn.add(Dense(32, 7))
# nn.add(Softmax())


# nn.compile()
# save(nn, "src/models/test")
# cProfile.run("train(nn, data_in, data_out, 10, batch_size = 1024, learning_rate = 0.1)")
