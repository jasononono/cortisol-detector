import numpy as np
import time


def categorial_cross_entropy(activation, expected):
    return -np.sum(expected * np.log(activation), axis = -1)

def compute_gradient(activation, expected):
    return activation - expected

def batched(model, data_in, data_out, epochs, learning_rate = 0.01, batch_size = 256,
            valid_in = None, valid_out = None, valid_size = None):
    batch_size = min(batch_size, data_in.shape[0])
    valid_in = data_in if valid_in is None else valid_in
    valid_out = data_out if valid_out is None else valid_out
    valid_size = min(valid_size or batch_size, valid_in.shape[0])
    valid_set = np.random.permutation(valid_in.shape[0])[:valid_size]

    model.compile()
    output = model.forward(valid_in[valid_set])
    loss = np.mean(categorial_cross_entropy(output, valid_out[valid_set]))
    time_begin = time.perf_counter()
    print(f"Training for {epochs} epochs\t\tinitial loss = {loss:.04f}\n")

    for e in range(epochs):
        time_start = time.perf_counter()
        samples = np.random.permutation(data_in.shape[0])
        samples_in = data_in[samples]
        samples_out = data_out[samples]

        for i in range(0, data_in.shape[0], batch_size):
            gradient = compute_gradient(model.forward(samples_in[i:i + batch_size]), samples_out[i:i + batch_size])
            model.backpropagate(gradient, learning_rate / batch_size)

        output = model.forward(valid_in[valid_set])
        loss = np.mean(categorial_cross_entropy(output, valid_out[valid_set]))
        time_end = time.perf_counter()
        print(f"epoch {e + 1}/{epochs}\t\tloss = {loss:.04f}\t\ttime = {time_end - time_start:.04f}\t\telapsed = {time_end - time_begin:.04f}")
