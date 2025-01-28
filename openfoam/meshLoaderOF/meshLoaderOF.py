import sys
sys.path.append('../')
import torch
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam')
sys.path.insert(0,'/home/maoy/data/OpenFOAM/maoy-6/solver/repo/fluidfoam/test')

from fluidfoam import readmesh, readscalar, readvector, OpenFoamFile, getVolumes
from openfoam_file_loader import readScalarVolType, readScalarSurfaceType, readVectorVolType, readVectorSurfaceType
import numpy as np

class meshLoaderOF:
    def __init__(self,sol,rank,dtype=torch.float32):

        c, vols = getVolumes(sol,verbose=False)

        total_vol_array_size = 0
        # internal cells
        xs,ys,zs = readmesh(sol,verbose=False)
        n_cells = len(vols)
        
        total_vol_array_size += n_cells
        total_boundary_face_size = 0
        
        neighfile = OpenFoamFile(sol+"/constant/polyMesh",name="neighbour",verbose=False)
        n_internal_faces = neighfile.nb_faces
        
        # boundary mesh
        bounfile = OpenFoamFile(sol+"/constant/polyMesh",name="boundary",verbose=False)
        boundary_names = list(bounfile.boundaryface.keys())
        dict_boundary_start_face_surface_field = {}
        dict_boundary_end_face_surface_field = {}
        
        dict_boundary_start_face_vol_field = {}
        dict_boundary_end_face_vol_field = {}
        
        for name in boundary_names:
            key = name.decode('utf-8')
            xs,ys,zs = readmesh(sol,boundary=key)
            n_start_face = int(bounfile.boundaryface[str.encode(key)][b'startFace'])
            n_faces = int(bounfile.boundaryface[str.encode(key)][b'nFaces'])
            total_vol_array_size += n_faces
            total_boundary_face_size += n_faces
            print(f'{key}: {len(xs)} {len(ys)} {len(zs)}, startFace:{n_start_face}, nFaces:{n_faces}')
            dict_boundary_start_face_vol_field[key] = n_start_face - n_internal_faces + n_cells
            dict_boundary_end_face_vol_field[key] = n_start_face + n_faces - n_internal_faces + n_cells
        
            dict_boundary_start_face_surface_field[key] = n_start_face
            dict_boundary_end_face_surface_field[key] = n_start_face + n_faces
        
        ownerfile = OpenFoamFile(sol+"/constant/polyMesh",name="owner",verbose=False)
        
        total_surface_array_size = len(ownerfile.values)
        
        Cx = np.zeros(total_vol_array_size)
        Cy = np.zeros(total_vol_array_size)
        Cz = np.zeros(total_vol_array_size)
        readVectorVolType(Cx,Cy,Cz, \
                n_cells, \
                bounfile, \
                "constant/polyMesh", \
                sol, \
                "C", \
                owners=ownerfile.values,\
                neighbors=neighfile.values,\
                bc_names=[]\
                )
        
        # internal field of magSf
        magSf = np.zeros(total_surface_array_size)
        readScalarSurfaceType(magSf, n_internal_faces, bounfile, "constant/polyMesh", sol, "magSf")
        
        weights = np.zeros(total_surface_array_size)
        readScalarSurfaceType(weights, n_internal_faces, bounfile, "constant", sol, "weights")
        
        deltaCoeffs = np.zeros(total_surface_array_size)
        readScalarSurfaceType(deltaCoeffs, n_internal_faces, bounfile, "constant", sol, "deltaCoeffs")
        
        ########################readVectorSurfaceType#############################
        Sfx = np.zeros(total_surface_array_size)
        Sfy = np.zeros(total_surface_array_size)
        Sfz = np.zeros(total_surface_array_size)
        readVectorSurfaceType(Sfx,Sfy,Sfz, n_internal_faces, bounfile, "constant/polyMesh", sol, "S")
        
        Cfx = np.zeros(total_surface_array_size)
        Cfy = np.zeros(total_surface_array_size)
        Cfz = np.zeros(total_surface_array_size)
        readVectorSurfaceType(Cfx,Cfy,Cfz, n_internal_faces, bounfile, "constant/polyMesh", sol, "Cf")
        
        deltax = np.zeros(total_surface_array_size)
        deltay = np.zeros(total_surface_array_size)
        deltaz = np.zeros(total_surface_array_size)
        readVectorSurfaceType(deltax,deltay,deltaz, n_internal_faces, bounfile, "constant/polyMesh", sol, "delta")

        self.Cx = torch.from_numpy(Cx).to(dtype).to(rank)
        self.Cy = torch.from_numpy(Cy).to(dtype).to(rank)
        self.Cz = torch.from_numpy(Cz).to(dtype).to(rank)

        self.magSf = torch.from_numpy(magSf).to(dtype).to(rank)
        self.weights = torch.from_numpy(weights).to(dtype).to(rank)
        self.deltaCoeffs = torch.from_numpy(deltaCoeffs).to(dtype).to(rank)

        self.Sfx = torch.from_numpy(Sfx).to(dtype).to(rank)
        self.Sfy = torch.from_numpy(Sfy).to(dtype).to(rank)
        self.Sfz = torch.from_numpy(Sfz).to(dtype).to(rank)

        self.Cfx = torch.from_numpy(Cfx).to(dtype).to(rank)
        self.Cfy = torch.from_numpy(Cfy).to(dtype).to(rank)
        self.Cfz = torch.from_numpy(Cfz).to(dtype).to(rank)

        self.deltax = torch.from_numpy(deltax).to(dtype).to(rank)
        self.deltay = torch.from_numpy(deltay).to(dtype).to(rank)
        self.deltaz = torch.from_numpy(deltaz).to(dtype).to(rank)

        self.V = torch.from_numpy(vols).to(dtype).to(rank)
        self.neighbors = torch.from_numpy(neighfile.values).to(rank)
        self.owners = torch.from_numpy(ownerfile.values).to(rank)

        self.n_cells = n_cells
        self.bounfile = bounfile
        self.n_bound_faces = total_boundary_face_size
        self.n_internal_faces = n_internal_faces
        self.sol = sol
        self.voltype_size = total_vol_array_size
        self.surfacetype_size = total_surface_array_size

        self.boundary_start_face_surface_field = dict_boundary_start_face_surface_field
        self.boundary_end_face_surface_field = dict_boundary_end_face_surface_field
        
        self.boundary_start_face_vol_field = dict_boundary_start_face_vol_field
        self.boundary_end_face_vol_field = dict_boundary_end_face_vol_field

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = '../Burger1D/configs/custom/burger1D.yaml'
    config = load_config(config_file)
    sol = config['data']['sol']
    loader = meshLoaderOF(sol,rank)
