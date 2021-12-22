import os

# Check if folder exists, if not, create one
def check_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


if __name__ == "__main__":
    pass
