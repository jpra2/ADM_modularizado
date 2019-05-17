from definitions.mesh_manager import MeshManager
from definitions.dual_primal import DualPrimal
import time
import os
import shutil
import yaml
import numpy as np

__all__ = ['dualprimal', 'MM', 'data_loaded']

t0 = time.time()
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')

### deletar o conteudo das pastas no diretorio flying
try:
    shutil.rmtree(flying_dir)
except:
    pass
try:
    shutil.rmtree(mono_dir)
except:
    pass
try:
    shutil.rmtree(bif_dir)
except:
    pass
os.makedirs(flying_dir)
os.makedirs(mono_dir)
os.makedirs(bif_dir)
################################
tags_criadas_aqui = []

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
        # data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
        # data_loaded = yaml.full_load(stream)

input_file = data_loaded['input_file']
ext_msh = input_file + '.msh'
MM = MeshManager(ext_msh)
os.chdir(flying_dir)
with open('__init__.py', 'w') as f:
    pass

all_volumes=MM.all_volumes
all_centroids = MM.all_centroids
MM.mb.tag_set_data(MM.cent_tag, all_volumes, MM.all_centroids)

verts = MM.mb.get_connectivity(all_volumes[0])
coords = MM.mb.get_coords(verts).reshape([len(verts), 3])
mins = coords.min(axis=0)
maxs = coords.max(axis=0)
dx0, dy0, dz0 = maxs - mins
lx, ly, lz = maxs - mins

verts = MM.mb.get_connectivity(all_volumes)
coords = MM.mb.get_coords(verts).reshape([len(verts), 3])
mins = coords.min(axis=0)
maxs = coords.max(axis=0)
Ltotal = maxs - mins + 1e-9 ### correcao de ponto flutuante
Lx, Ly, Lz = Ltotal

# nx, ny, nz = (Ltotal/(maxs - mins)).astype(np.int32)
nx, ny, nz = (Ltotal/(np.array([lx, ly, lz])) + 1e-9).astype(np.int32)

# Distância, em relação ao poço, até onde se usa malha fina
r0 = data_loaded['rs']['r0']
# Distância, em relação ao poço, até onde se usa malha intermediária
r1 = data_loaded['rs']['r1']

l1=data_loaded['Ls']['L1']
l2=data_loaded['Ls']['L2']

print("")
print("INICIOU PRÉ PROCESSAMENTO")
t1 = time.time()
dualprimal = DualPrimal(MM, Lx, Ly, Lz, mins, l2, l1, dx0, dy0, dz0, lx, ly, lz)
