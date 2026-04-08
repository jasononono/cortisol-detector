import json, sys, os


def path(name):
    try:
        directory = sys._MEIPASS
    except AttributeError:
        return name
    return os.path.join(directory, name)

json_cache = {}

def load_json(file_name):
    if file_name in json_cache:
        return json_cache[file_name]

    with open(path(file_name), 'r') as file:
        data = json.load(file)
    json_cache[file_name] = data
    return data