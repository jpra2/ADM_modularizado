import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import yaml
import sys
from utils import pymoab_utils as utpy

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')


os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
        # data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
        # data_loaded = yaml.full_load(stream)

input_file = data_loaded['input_file']
ext_h5m = input_file + '_dual_primal.h5m'
os.chdir(flying_dir)

mesh_file = ext_h5m
mb = core.Core()
root_set = mb.get_root_set()
mtu = topo_util.MeshTopoUtil(mb)
mb.load_file(mesh_file)

all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)
cent_tag = mb.tag_get_handle('CENT')
all_centroids = mb.tag_get_data(cent_tag, all_volumes)
press_value_tag = mb.tag_get_handle('P')
vazao_value_tag = mb.tag_get_handle('Q')
wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)

wells_injector = mb.create_meshset()
wells_producer = mb.create_meshset()
mb.tag_set_data(wells_injector_tag, 0, wells_injector)
mb.tag_set_data(wells_producer_tag, 0, wells_producer)

bifasico = data_loaded['bifasico']

if bifasico:
    ############################################################
    ###tags_do_bifasico
    # mi_w_tag = mb.tag_get_handle('MI_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # mi_o_tag = mb.tag_get_handle('MI_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # gama_w_tag = mb.tag_get_handle('GAMA_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # gama_o_tag = mb.tag_get_handle('GAMA_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # sor_tag = mb.tag_get_handle('SOR', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # swc_tag = mb.tag_get_handle('SWC', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # nw_tag = mb.tag_get_handle('NW', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    # no_tag = mb.tag_get_handle('NO', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    sat_tag = mb.tag_get_handle('SAT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    fw_tag = mb.tag_get_handle('FW', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lamb_w_tag = mb.tag_get_handle('LAMB_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lamb_o_tag = mb.tag_get_handle('LAMB_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lbt_tag = mb.tag_get_handle('LBT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    total_flux_tag = mb.tag_get_handle('TOTAL_FLUX', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    flux_w_tag = mb.tag_get_handle('FLUX_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mobi_in_faces_tag = mb.tag_get_handle('MOBI_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mobi_w_in_faces_tag = mb.tag_get_handle('MOBI_W_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    flux_in_faces_tag = mb.tag_get_handle('FLUX_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    loops_tag = mb.tag_get_handle('LOOPS', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    fw_in_faces_tag = mb.tag_get_handle('FW_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    dfds_tag = mb.tag_get_handle('DFDS', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mi = mi = data_loaded['dados_monofasico']['mi']
    gama = data_loaded['dados_monofasico']['gama']
    ############################################################

else:
    mi = data_loaded['dados_monofasico']['mi']
    gama = data_loaded['dados_monofasico']['gama']

############################################################


verts = mb.get_connectivity(all_volumes)
coords = mb.get_coords(verts).reshape([len(verts), 3])
mins = coords.min(axis=0)
maxs = coords.max(axis=0)
Ltotal = maxs - mins + 1e-9 ### correcao de ponto flutuante
Lx, Ly, Lz = Ltotal

gravity = data_loaded['gravity']

inds_wells = []
for well in data_loaded['Wells_structured']:
    w = data_loaded['Wells_structured'][well]
    if w['type_region'] == 'box':
        box_volumes = np.array([np.array(w['region1']), np.array(w['region2'])])
        inds0 = np.where(all_centroids[:,0] > box_volumes[0,0])[0]
        inds1 = np.where(all_centroids[:,1] > box_volumes[0,1])[0]
        inds2 = np.where(all_centroids[:,2] > box_volumes[0,2])[0]
        c1 = set(inds0) & set(inds1) & set(inds2)
        inds0 = np.where(all_centroids[:,0] < box_volumes[1,0])[0]
        inds1 = np.where(all_centroids[:,1] < box_volumes[1,1])[0]
        inds2 = np.where(all_centroids[:,2] < box_volumes[1,2])[0]
        c2 = set(inds0) & set(inds1) & set(inds2)
        inds_vols = list(c1 & c2)
        inds_wells += inds_vols
        volumes = rng.Range(np.array(all_volumes)[inds_vols])
    else:
        raise NameError("Defina o tipo de regiao em type_region: 'box'")

    value = float(w['value'])

    if w['type_prescription'] == 'dirichlet':
        if gravity == False:
            pressao = np.repeat(value, len(volumes))

        else:
            z_elems_d = -1*mb.tag_get_data(cent_tag, volumes)[:,2]
            delta_z = z_elems_d + Lz
            pressao = gama*(delta_z) + value

        mb.tag_set_data(press_value_tag, volumes, pressao)

    elif  w['type_prescription'] == 'neumann':
        value = value/len(volumes)
        if w['type_well'] == 'injector':
            value = -value
        mb.tag_set_data(vazao_value_tag, volumes, np.repeat(value, len(volumes)))

    else:
        raise NameError("type_prescription == 'neumann' or 'dirichlet'")

    if w['type_well'] == 'injector':
        mb.add_entities(wells_injector, volumes)
    else:
        mb.add_entities(wells_producer, volumes)
