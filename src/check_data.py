import os

path = "data/train"
if os.path.exists(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            print(f"{folder} : {len(os.listdir(folder_path))} images")
else:
    print(f"Directory {path} not found.")