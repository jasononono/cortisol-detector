import numpy as np
import time


def categorial_cross_entropy(activation, expected):
    return -np.sum(expected * np.log(activation + 0.000000001), axis = -1)

def compute_gradient(activation, expected):
    return activation - expected

def get_loss(model, data_in, data_out):
    output = model.forward(data_in, data_in.shape[0])
    return np.mean(categorial_cross_entropy(output, data_out))

def get_accuracy(model, data_in, data_out):
    output = model.forward(data_in, data_in.shape[0])
    return np.sum(output * data_out) / data_in.shape[0] * 100

def batched(model, data_in, data_out, epochs, learning_rate = 0.01, batch_size = 256,
            target_loss = None, target_accuracy = None,
            valid_in = None, valid_out = None, valid_size = None):
    train_size = data_in.shape[0]
    batch_size = min(batch_size, train_size)
    valid_in = data_in if valid_in is None else valid_in
    valid_out = data_out if valid_out is None else valid_out
    valid_size = min(valid_size or valid_in.shape[0], valid_in.shape[0])
    valid_set = np.random.permutation(valid_in.shape[0])[:valid_size]

    model.compile()
    output = model.forward(valid_in[valid_set], valid_size)
    loss = np.mean(categorial_cross_entropy(output, valid_out[valid_set]))
    time_begin = time.perf_counter()
    print(f"Training for {epochs} epochs\t\tinitial loss = {loss:.04f}\n")

    try:

        for e in range(epochs):
            time_start = time.perf_counter()
            samples = np.random.permutation(data_in.shape[0])
            samples_in = data_in[samples]
            samples_out = data_out[samples]

            print('_' * (train_size // batch_size))
            for i in range(0, train_size - batch_size + 1, batch_size):
                gradient = compute_gradient(model.forward(samples_in[i:i + batch_size], batch_size), samples_out[i:i + batch_size])
                model.backpropagate(gradient, learning_rate / batch_size, batch_size)
                print('#', end = "")
            print()

            output = model.forward(valid_in[valid_set], valid_size)
            loss = np.mean(categorial_cross_entropy(output, valid_out[valid_set]))
            accuracy = np.sum(output * valid_out[valid_set]) / valid_size * 100
            if target_loss is not None and loss <= target_loss:
                print("Training stopped (Loss target reached)")
                break
            if target_accuracy is not None and accuracy >= target_accuracy:
                print("Training stopped (Accuracy target reached)")
                break

            time_end = time.perf_counter()
            print(f"epoch {e + 1}/{epochs}\t\tloss = {loss:.04f}\t\taccuracy = {accuracy:.02f}%\t\ttime = {time_end - time_start:.04f}s\t\telapsed = {time_end - time_begin:.02f}s")

        else:
            print("training stopped (Epochs completed)")

    except KeyboardInterrupt:
        print("\nTraining stopped (Terminated)")

    output = model.forward(valid_in[valid_set], valid_size)
    loss = np.mean(categorial_cross_entropy(output, valid_out[valid_set]))
    accuracy = np.sum(output * valid_out[valid_set]) / valid_size * 100
    print(f"loss = {loss:.04f}\t\taccuracy = {accuracy:.02f}%")