import os
import re
import sys
sys.path.append('../')
import torch
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam/test')

from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes
from openfoam_file_loader import readScalarVolType, readScalarSurfaceType, readVectorVolType, readVectorSurfaceType
import numpy as np
from meshLoaderOF.meshLoaderOF import meshLoaderOF

class porousFieldLoaderOF:
    def __init__(self,
                 mesh: meshLoaderOF,
                 rank: int,
                 dtype = torch.float32,
                 bc_names=[]):

        self.mesh = mesh
        sol = mesh.sol
        bounfile = mesh.bounfile
        n_bound_faces = mesh.n_bound_faces
        n_cells = mesh.n_cells
        n_internal_faces = mesh.n_internal_faces

        total_vol_array_size = n_cells + n_bound_faces

        numeric_filter = re.compile(r'^\d*\.?\d+$')
        filtered = [item for item in os.listdir(sol) if numeric_filter.match(item)]
        # Convert to float and sort in ascending order
        sorted_filtered = sorted(filtered, key=float)
        n_times = len(sorted_filtered)
        n_fields = 5 # Ux, Uy, p_rgh, T, alpha (Uz is not included as it is a 2D case)
        data = np.zeros((total_vol_array_size,n_times,n_fields))
        data_dict = []

        self.n_times = n_times
        for index, time in enumerate(sorted_filtered):
            #print(time)
            Ux = np.zeros(total_vol_array_size)
            Uy = np.zeros(total_vol_array_size)
            Uz = np.zeros(total_vol_array_size)
            field_name = 'U'
            if index == 0:
                field_name = 'U0'
            readVectorVolType(Ux, \
                    Uy, \
                    Uz, \
                    n_cells, \
                    bounfile, \
                    time, \
                    sol, \
                    field_name,\
                    owners=mesh.owners,\
                    neighbors=mesh.neighbors,\
                    bc_names=bc_names\
                    )
            data[:,index,0] = Ux
            data[:,index,1] = Uy
            #data[:,index,2] = Uz

            p_rgh = np.zeros(total_vol_array_size)
            T = np.zeros(total_vol_array_size)
            alpha = np.zeros(total_vol_array_size)

            readScalarVolType(p_rgh, \
                    n_cells, \
                    bounfile, \
                    time, \
                    sol, \
                    "p_rgh",\
                    owners=mesh.owners,\
                    neighbors=mesh.neighbors,\
                    bc_names=bc_names\
                    )
            data[:,index,2] = p_rgh

            readScalarVolType(T, \
                    n_cells, \
                    bounfile, \
                    time, \
                    sol, \
                    "T",\
                    owners=mesh.owners,\
                    neighbors=mesh.neighbors,\
                    bc_names=bc_names\
                    )
            data[:,index,3] = T

            readScalarVolType(alpha, \
                    n_cells, \
                    bounfile, \
                    time, \
                    sol, \
                    "lf",\
                    owners=mesh.owners,\
                    neighbors=mesh.neighbors,\
                    bc_names=bc_names\
                    )
            data[:,index,4] = alpha
        
        # Print tensor in scientific notation
        #np.set_printoptions(formatter={'float': '{:0.30g}'.format})
        #print(data[:n_cells,0,0])
        self.data = torch.from_numpy(data).to(dtype).to(rank)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = '../porousmelting/configs/custom/porousMelting.yaml'
    config = load_config(config_file)
    sol = config['data']['sol']
    bc_names=['defaultFaces']
    mesh = meshLoaderOF(sol,device)
    field = porousFieldLoaderOF(mesh,device,bc_names=bc_names)
