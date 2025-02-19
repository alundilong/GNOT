import sys
sys.path.append('../')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam/test')

import os
import torch
import numpy as np
import pickle

class openfoam_data_loader:
    def __init__(self, root_dir, dt, duration, rank, dtype=torch.float32, bc_names=[], max_cases=2):

        entries = os.listdir(root_dir)
        # Filter entries to include only directories
        directories = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry))]

        self.directories = []
        for i, directory in enumerate(directories):
            if max_cases and i + 1 > max_cases:
                break
            path = os.path.abspath(os.path.join(root_dir, directory))
            print(f'{i+1} {path}')
            self.directories.append(path)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bc_names=['front','back']

    case_tag = "long"
    dt = 200
    duration = 2000
    root_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/runs_time10_res50/'
    pickle_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res50/'
    if case_tag == "long":
        dt = 10
        root_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/runs_long_time10_res50/'
        pickle_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_long_time10_res50/'

    loader = openfoam_data_loader(root_dir,dt,duration,rank=device,bc_names=bc_names, max_cases=None)
    dir_to_store = os.path.join(pickle_dir, 'directories.pkl') 
    pickle.dump(loader.directories, open(dir_to_store,'wb'))

