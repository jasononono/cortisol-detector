import json, os

with open("src/app/info.json", 'r') as file:
    info = json.load(file)

dist_name = f"cortisol-detector"
command = f"""pyinstaller -y src/app/main.py \\
    --name {dist_name} \\
    --specpath spec/ \\
    --log-level WARN \\
    """

for file in info["data"]:
    d = os.path.dirname(file)
    command += f"--add-data ../src/app/{file}:{d if len(d) > 0 else '.'} "


os.system(command)

with open("cmd/dist.txt", 'w') as file:
    file.write(dist_name)

print(dist_name)