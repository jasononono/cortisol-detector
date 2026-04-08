from PIL import Image
import numpy as np
import file_util


data_info = file_util.load_json("data.json")
labels = data_info["labels"]
classes = {l: i for i, l in enumerate(labels)}


def load_set(name):
    raw = []
    expected = []

    for l in labels:
        amount = data_info[name][l]
        for i in range(amount):
            image = Image.open(f"{data_info['source-directory']}/{name}/{l}/im{i}.png")
            raw.append(np.array(image) / 255)
            expected.append(classes[l])

    data_input = np.array(raw)
    data_output = np.zeros((data_info[f"{name}-total"], data_info["labels-total"]))
    data_output[np.arange(data_info[f"{name}-total"]), expected] = 1

    return data_input, data_output


train_input, train_output = load_set("train")
test_input, test_output = load_set("test")
np.save("src/data/train_input.npy", train_input)
np.save("src/data/train_output.npy", train_output)
np.save("src/data/test_input.npy", test_input)
np.save("src/data/test_output.npy", test_output)