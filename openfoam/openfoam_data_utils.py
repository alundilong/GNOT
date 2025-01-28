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

class openfoam_data_loader:
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
        nCells = len(Cx)
        # nf includes (x, time, U0)
        a_nfield = 10 # x,y,z,time,U0,p_rgh,T,alpha,m
        n_total = nCells*nt
        a_data = torch.zeros((n_total,a_nfield),dtype=dtype)
        # input data has bs, nCells,ny,nz, nt, nf
        a_data[:,0] = Cx.repeat(nt)
        a_data[:,1] = Cy.repeat(nt)
        a_data[:,2] = Cz.repeat(nt)
        time = torch.linspace(0,duration,nt,dtype=dtype)
        a_data[:,3] = time.repeat(nCells)
        a_data[:,4] = U0x.repeat(nt)
        a_data[:,5] = U0y.repeat(nt)
        #a_data[:,:,:,6] = U0z.reshape(1,nCells,ny,nz,1).repeat(bz,1,1,1,nt)
        a_data[:,6] = p_rgh0.repeat(nt)
        a_data[:,7] = T0.repeat(nt)
        a_data[:,8] = alpha0.repeat(nt)
        a_data[:,9] = mask0.repeat(nt)

        u_nfield = 5 # U,p_rgh,T,alpha
        u_data = torch.zeros((n_total,u_nfield),dtype=dtype)
        #u_data[:,:,:,0] = U0.reshape(1,nCells,1).repeat(bz,1,nt)
        field = porousFieldLoaderOF(mesh,rank,bc_names=bc_names)
        #print(field.data[:,0,0])
        u_data[:,:] = field.data.reshape(-1,u_nfield)
        
        self.X = a_data
        self.Y = u_data
        self.mesh = mesh
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bc_names=['defaultFaces']
    path = '/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/runs/max_16_min_11_points_8/'
    dt = 10
    duration = 1000
    data = openfoam_data_loader(path,dt,duration,rank=device,bc_names=bc_names)

