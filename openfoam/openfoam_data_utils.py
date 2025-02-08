import sys
sys.path.append('../')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam/test')

import os
import torch
import numpy as np
from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes
from openfoam_file_loader import readScalarVolType, readScalarSurfaceType, readVectorVolType, readVectorSurfaceType
from openfoam_file_writer import writeScalarVolType, writeVectorVolType
from meshLoaderOF.meshLoaderOF import meshLoaderOF
from fieldLoaderOF.porousFieldLoaderOF import porousFieldLoaderOF

import pickle

class openfoam_data_single_case_loader:
    def __init__(self, path, dt, duration, rank, dtype=torch.float32, bc_names=[]):
        sol = path 
        nt = int(duration/dt)+1
        mesh = meshLoaderOF(sol,rank,dtype=dtype)
        n_bound_faces = mesh.n_bound_faces
        n_cells = mesh.n_cells
        Ux = np.zeros(n_cells+n_bound_faces)
        Uy = np.zeros(n_cells+n_bound_faces)
        Uz = np.zeros(n_cells+n_bound_faces)
        time = '0'
        bounfile = mesh.bounfile
        readVectorVolType(Ux, \
                Uy, \
                Uz, \
                n_cells, \
                bounfile, \
                time, \
                sol, \
                "U0",
                owners=mesh.owners,\
                neighbors=mesh.neighbors,\
                bc_names=bc_names\
                )
        U0x = torch.from_numpy(Ux).to(rank)
        U0y = torch.from_numpy(Uy).to(rank)
        #U0z = torch.from_numpy(Uz).to(rank)

        p_rgh = np.zeros(n_cells+n_bound_faces)
        T = np.zeros(n_cells+n_bound_faces)
        alpha = np.zeros(n_cells+n_bound_faces)
        mask = np.zeros(n_cells+n_bound_faces)
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
        temp_bc_names = bc_names.copy()
        temp_bc_names.append('floor')
        temp_bc_names.append('ceiling')
        readScalarVolType(T, \
                n_cells, \
                bounfile, \
                time, \
                sol, \
                "T",\
                owners=mesh.owners,\
                neighbors=mesh.neighbors,\
                bc_names=temp_bc_names\
                )
        readScalarVolType(alpha, \
                n_cells, \
                bounfile, \
                time, \
                sol, \
                "alpha",\
                owners=mesh.owners,\
                neighbors=mesh.neighbors,\
                bc_names=bc_names\
                )
        readScalarVolType(mask, \
                n_cells, \
                bounfile, \
                time, \
                sol, \
                "mask",\
                owners=mesh.owners,\
                neighbors=mesh.neighbors,\
                bc_names=bc_names\
                )
        p_rgh0 = torch.from_numpy(p_rgh).to(rank)
        T0 = torch.from_numpy(T).to(rank)
        alpha0 = torch.from_numpy(alpha).to(rank)
        mask0 = torch.from_numpy(mask).to(rank)

        Cx = mesh.Cx
        Cy = mesh.Cy
        Cz = mesh.Cz
        nCoordinate = len(Cx)
        # nf includes (x, time, U0)
        a_nfield = 9 # x,y,time,U0,p_rgh,T,alpha,m
        n_total = nCoordinate*nt
        a_data = torch.zeros((n_total,a_nfield),dtype=dtype)
        # input data has bs, nCoordinate,ny,nz, nt, nf
        a_data[:,0] = Cx.repeat(nt)
        a_data[:,1] = Cy.repeat(nt)
        #a_data[:,2] = Cz.repeat(nt)
        time = torch.linspace(0,duration,nt,dtype=dtype)
        a_data[:,2] = time.repeat(nCoordinate)
        a_data[:,3] = U0x.repeat(nt)
        a_data[:,4] = U0y.repeat(nt)
        #a_data[:,:,:,6] = U0z.reshape(1,nCoordinate,ny,nz,1).repeat(bz,1,1,1,nt)
        a_data[:,5] = p_rgh0.repeat(nt)
        a_data[:,6] = T0.repeat(nt)
        a_data[:,7] = alpha0.repeat(nt)
        a_data[:,8] = mask0.repeat(nt)

        u_nfield = 5 # U,p_rgh,T,alpha
        u_data = torch.zeros((n_total,u_nfield),dtype=dtype)
        #u_data[:,:,:,0] = U0.reshape(1,nCoordinate,1).repeat(bz,1,nt)
        field = porousFieldLoaderOF(mesh,rank,bc_names=bc_names)
        #print(field.data[:,0,0])
        u_data[:,:] = field.data.reshape(-1,u_nfield)
        
        self.X = a_data
        self.Y = u_data
        self.mesh = mesh
        
        remove_range = []
        for key in bc_names:
            start = mesh.boundary_start_face_vol_field[key]
            end = mesh.boundary_end_face_vol_field[key]
            remove_range.append((start,end))

        indices_to_keep = np.ones(n_total, dtype=bool)
        indices_to_keep_bc = np.ones(n_total, dtype=bool)

        for start, end in remove_range:
            for i in range(nt):
                indices_to_keep[i*nCoordinate + start:i*nCoordinate + end] = False
                indices_to_keep_bc[i*nCoordinate + start:i*nCoordinate + end] = False
        for i in range(nt):
            indices_to_keep_bc[i*nCoordinate:i*nCoordinate+n_cells] = False
        self.boundary_coordinates = self.X[indices_to_keep_bc,:3] # x,y,t
        self.X = self.X[indices_to_keep,:]
        self.Y = self.Y[indices_to_keep,:]

class openfoam_data_loader:
    def __init__(self, root_dir, dt, duration, rank, dtype=torch.float32, bc_names=[], max_cases=2):

        entries = os.listdir(root_dir)
        # Filter entries to include only directories
        directories = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry))]

        self.data_all = []
        for i, directory in enumerate(directories):
            if max_cases and i + 1 > max_cases:
                break
            path = os.path.abspath(os.path.join(root_dir, directory))
            print(f'{i} {path}')
            loader = openfoam_data_single_case_loader(path,dt,duration,rank,bc_names=bc_names)
            single = []
            single.append(loader.X.numpy())
            single.append(loader.Y.numpy())
            single.append(np.array([0.0]*6))
            single.append([loader.boundary_coordinates.numpy()])
            self.data_all.append(single)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bc_names=['front','back']
    path = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/runs/max_16_min_11_points_8/'
    dt = 10
    duration = 2000
    #single_loader = openfoam_data_single_case_loader(path,dt,duration,rank=device,bc_names=bc_names)

    root_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/runs_long_time10_res50/'

    pickle_dir = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_long_time10_res50/'

    # Check if the directory exists
    if not os.path.exists(pickle_dir):
        # Create the directory
        os.makedirs(pickle_dir)
        print(f"Directory '{pickle_dir}' created.")
    else:
        print(f"Directory '{pickle_dir}' already exists.")
    loader = openfoam_data_loader(root_dir,dt,duration,rank=device,bc_names=bc_names, max_cases=None)
    
    print(f'total sample: {len(loader.data_all)}')

    dir_to_store = os.path.join(pickle_dir, 'porousmelting_train.pkl') 
    pickle.dump(loader.data_all, open(dir_to_store,'wb'))

