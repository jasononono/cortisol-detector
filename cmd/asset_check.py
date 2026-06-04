import json, pathlib, fnmatch


def match(file, patterns):
    for p in patterns:
        if file.match(p):
            return True
    return False

def format_file(wd, file):
    wd += '/'
    file_name = str(file)
    if file_name[:len(wd)] == wd:
        file_name = file_name[len(wd):]

    if file.is_dir():
        file_name += '/'
    return file_name

def load_data(wd, ignore, directory = ""):
    path = pathlib.Path(f"{wd}/{directory}")
    data = []

    for file in path.iterdir():
        if match(file, ignore):
            continue

        name = format_file(wd, file)
        if file.is_dir():
            data.extend(load_data(wd, ignore, name))
        else:
            data.append(name)

    return data


with open("info.json", 'r') as file:
    ignore = json.load(file)["data-ignore"]
with open("src/app/info.json", 'r') as file:
    info = json.load(file)

info["data"] = load_data("src/app", ignore)

with open("src/app/info.json", 'w') as file:
    json.dump(info, file, indent = 2)