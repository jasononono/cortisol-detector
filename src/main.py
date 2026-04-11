import numpy as np
import model, training
from layer import Convolution, ReLU, Reshape, Dense, Softmax, MaxPool


data_in = np.load("src/data/train_input.npy")
data_out = np.load("src/data/train_output.npy")
valid_in = np.load("src/data/test_input.npy")
valid_out = np.load("src/data/test_output.npy")
data_in = data_in.reshape(-1, 1, 48, 48)
valid_in = valid_in.reshape(-1, 1, 48, 48)


nn = model.Model()

nn.add(Convolution(48, 3, 1, 64, 1))
nn.add(ReLU())
nn.add(Convolution(46, 3, 64, 64, 1))
nn.add(ReLU())
nn.add(MaxPool(44, 64, 2))

nn.add(Convolution(22, 3, 64, 128, 1))
nn.add(ReLU())
nn.add(MaxPool(20, 128, 2))

nn.add(Convolution(10, 3, 128, 256, 1))
nn.add(ReLU())
nn.add(MaxPool(8, 256, 2))

nn.add(Reshape((256, 4, 4), 4096))
nn.add(Dense(4096, 128))
nn.add(ReLU())
nn.add(Dense(128, 7))
nn.add(Softmax())


training.batched(nn, data_in, data_out, 10, batch_size = 256,
                 target_loss = 0.05, target_accuracy = 95, learning_rate = 0.0005,
                 valid_in = valid_in, valid_out = valid_out, valid_size = 1000)
file_name = input("save model as (leave blank to discard) > ")
if file_name:
    model.save(nn, "src/models/" + file_name)


'''
import numpy as np
import cProfile
import model
from layer import Convolution, ReLU, Reshape, Dense, Softmax, MaxPool
import training



data_in = np.load("src/data_mnist/input_train.npy")
data_out = np.load("src/data_mnist/output_train.npy")
valid_in = np.load("src/data_mnist/input_valid.npy")
valid_out = np.load("src/data_mnist/output_valid.npy")
data_in = data_in.reshape(-1, 1, 28, 28)
valid_in = valid_in.reshape(-1, 1, 28, 28)


# nn = model.load("src/models/mnist")
# # display mode confusion
# import tensorflow as tf
# import seaborn as sb
# import matplotlib.pyplot as plt
# probability = nn.forward(valid_in[:1000], 1000)
# output_classes = probability.argmax(axis = -1)
# confusion_matrix = tf.math.confusion_matrix(valid_out[:1000].argmax(axis = -1), output_classes)
# fig = sb.heatmap(confusion_matrix, annot = True, fmt = 'g', cmap = "Greens")
# fig.set_xlabel("Predicted")
# fig.set_ylabel("True")
# fig.set_title("Confusion Matrix")
# fig.figure.set_size_inches(10, 10)
# plt.show()


# nn = model.Model()
# nn.add(Convolution(28, 5, 1, 32, 1))
# nn.add(ReLU())
# nn.add(MaxPool(24, 32, 2))
# nn.add(Convolution(12, 3, 32, 64, 1))
# nn.add(ReLU())
# nn.add(MaxPool(10, 64, 2))
# nn.add(Reshape((64, 5, 5), 1600))
# nn.add(Dense(1600, 10))
# nn.add(Softmax())
#
# training.batched(nn, data_in, data_out, 10, batch_size = 256,
#                  target_loss = 0.01, target_accuracy = 99.5,
#                  valid_in = valid_in, valid_out = valid_out)
# file_name = input("save model as (leave blank to discard) > ")
# if file_name:
#     model.save(nn, "src/models/" + file_name)


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
'''