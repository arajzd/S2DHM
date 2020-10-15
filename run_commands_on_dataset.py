import numpy as np
import os

from tempfile import mkstemp
from shutil import move
from os import fdopen, rename


def make_dataset_class(file_path, dataset_name):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if "class SfDataset(BaseDataset):" in line:
                    line = "class " + dataset_name + "(BaseDataset):\n"
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), dataset_name) + ".py")


def walklevel(path, depth=1):
    """It works just like os.walk, but you can pass it a level parameter
       that indicates how deep the recursion will go.
       If depth is -1 (or less than 0), the full depth is walked.
    """
    # if depth is negative, just walk
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs, files

    # path.count works because is a file has a "/" it will show up in the list
    # as a ":"
    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + depth <= num_sep_this:
            del dirs[:]


def make_dataset_gin_file(file_path, dataset_name):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:

                if "ROOT = " in line:
                    line = dataset_name.upper() + f"_ROOT = '{os.path.join(current_path, sub_folder)}/" + dataset_name + "'\n"
                elif ".image_folder =" in line:
                    line = dataset_name + ".image_folder = 'imageset/'\n"
                elif ".query_sequences =" in line:
                    line = dataset_name + ".query_sequences = ['query']\n"
                elif "nvm_model =" in line:
                    line = f"RobotCarDataset.nvm_model = '{os.path.join(current_path, sub_folder)}/" + dataset_name + "/reconstruction.nvm'\n"
                elif "triangulation_data_file" in line:
                    line = f"RobotCarDataset.triangulation_data_file = '{s2dhm_path}data/triangulation/" + dataset_name + ".npz'\n"

                line = line.replace("RobotCarDataset", dataset_name, 100)
                line = line.replace("robotcar_dataset", dataset_name, 100)
                line = line.replace('robotcar', dataset_name, 100)
                line = line.replace("ROBOTCAR", dataset_name.upper(), 100)
                line = line.replace("robotcar_dataset", dataset_name, 100)
                line = line.replace("overcast-reference", "db", 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), dataset_name) + ".gin")


def make_dataset_sparse_dense_gin_file(file_path, dataset_name):
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if not "weight" in line:
                    line = line.replace("robotcar", dataset_name, 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), "run_sparse_to_dense_on_" + dataset_name + ".gin"))


current_path = "/home/ara/Documents/Internship"
sub_folder = "OpenSfM/data/"
s2dhm_path = "/home/ara/PycharmProjects/HyperColumn/S2DHM/"
if __name__ == '__main__':

    for dirpath, dirnames, filenames in walklevel(os.path.join(current_path, sub_folder)):
        if not os.path.exists(os.path.join(dirpath, "colmap_models", "db_with_query", "images.txt")):
            continue
        dataset_name = dirpath.split("/")[-1]
        os.system(
            f"python3 {s2dhm_path}s2dhm/datasets/internal/read_model.py " + dirpath +
            "/colmap_models/db_triangulated .txt " +
            dataset_name)
        make_dataset_class(f"{s2dhm_path}s2dhm/datasets/sf_dataset.py", dataset_name)
        make_dataset_gin_file(f"{s2dhm_path}s2dhm/configs/datasets/robotcar.gin",
                              dataset_name)
        make_dataset_sparse_dense_gin_file(
            f"{s2dhm_path}s2dhm/configs/runs/run_sparse_to_dense_on_robotcar.gin",
            dataset_name)
        os.chdir(f"{s2dhm_path}s2dhm")
        os.system("python3 run.py "
                  "--dataset " + dataset_name + " --mode sparse_to_dense --log_images")
