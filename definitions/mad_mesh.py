import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import yaml
import sys
from utils import pymoab_utils as utpy
from definitions.functions1 import *
from utils.prolongation import *
from utils.others_utils import OtherUtils as oth
import sympy
from scipy.sparse import csc_matrix, vstack, find, linalg
import scipy

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')

tags_criadas_aqui = []

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

intern_adjs_by_dual = np.load('intern_adjs_by_dual.npy')
faces_adjs_by_dual = np.load('faces_adjs_by_dual.npy')

all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)
cent_tag = mb.tag_get_handle('CENT')
all_centroids = mb.tag_get_data(cent_tag, all_volumes)
press_value_tag = mb.tag_get_handle('P')
vazao_value_tag = mb.tag_get_handle('Q')
wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
tags_criadas_aqui += ['WELLS_INJECTOR', 'WELLS_PRODUCER']

wells_injector = mb.create_meshset()
wells_producer = mb.create_meshset()
mb.tag_set_data(wells_injector_tag, 0, wells_injector)
mb.tag_set_data(wells_producer_tag, 0, wells_producer)

bifasico = data_loaded['bifasico']
r0 = data_loaded['rs']['r0']
r1 = data_loaded['rs']['r1']

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
    tags_criadas_aqui += ['SAT', 'FW', 'LAMB_W', 'LAMB_O', 'LBT', 'TOTAL_FLUX', 'FLUX_W', 'MOBI_IN_FACES', 'MOBI_W_IN_FACES',
                          'FLUX_IN_FACES', 'LOOPS', 'FW_IN_FACES', 'DFDS']
    ############################################################

else:
    mi = data_loaded['dados_monofasico']['mi']
    gama = data_loaded['dados_monofasico']['gama']

############################################################


verts = mb.get_connectivity(all_volumes)
coords = mb.get_coords(verts).reshape([len(verts), 3])
mins = coords.min(axis=0)
maxs = coords.max(axis=0)
Ltotal = maxs - mins + 1e-9  # correcao de ponto flutuante
Lx, Ly, Lz = Ltotal

gravity = data_loaded['gravity']

bvd = []
bvn = []

volumes_d = []
volumes_n = []

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
        bvd.append(box_volumes)
        volumes_d += list(volumes)
        if gravity == False:
            pressao = np.repeat(value, len(volumes))

        else:
            z_elems_d = -1*mb.tag_get_data(cent_tag, volumes)[:,2]
            delta_z = z_elems_d + Lz
            pressao = gama*(delta_z) + value

        mb.tag_set_data(press_value_tag, volumes, pressao)

    elif  w['type_prescription'] == 'neumann':
        volumes_n += list(volumes)
        bvn.append(box_volumes)
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

np.save('bvd', bvd)
np.save('bvn', bvn)

bvfd = list()
for bvdi in bvd:
    bvfd.append(np.array([np.array([bvdi[0][0]-r0, bvdi[0][1]-r0, bvdi[0][2]-r0]), np.array([bvdi[1][0]+r0, bvdi[1][1]+r0, bvdi[1][2]+r0])]))
bvfd = np.array(bvfd)

bvfn = list()
for bvfni in bvn:
    bvfn.append(np.array([np.array([bvn[0][0]-r0, bvn[0][1]-r0, bvn[0][2]-r0]), np.array([bvn[1][0]+r0, bvn[1][1]+r0, bvn[1][2]+r0])]))
bvfn = np.array(bvfn)

# bvfd = np.array([np.array([bvd[0][0]-r0, bvd[0][1]-r0, bvd[0][2]-r0]), np.array([bvd[1][0]+r0, bvd[1][1]+r0, bvd[1][2]+r0])])
# bvfn = np.array([np.array([bvn[0][0]-r0, bvn[0][1]-r0, bvn[0][2]-r0]), np.array([bvn[1][0]+r0, bvn[1][1]+r0, bvn[1][2]+r0])])

bvid = list()
for bvidi in bvd:
    gg = np.array([bvidi[0]-r1, bvidi[1]+r1])
    bvid.append(gg)
bvid = np.array(bvid)

bvin = list()
for bvini in bvn:
    bvin.append(np.array([np.array([bvini[0][0]-r1, bvini[0][1]-r1, bvini[0][2]-r1]), np.array([bvini[1][0]+r1, bvini[1][1]+r1, bvini[1][2]+r1])]))
bvin = np.array(bvin)

# bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
# bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])

volumes_d = rng.Range(volumes_d)
volumes_n = rng.Range(volumes_n)

# volumes finos por neumann
volumes_fn = []
for i in bvfn:
    volumes_fn += list(get_box(all_volumes, all_centroids, i, False))
volumes_fn = rng.Range(volumes_fn)
# volumes_fn = get_box(all_volumes, all_centroids, bvfn, False)

# volumes finos por Dirichlet
volumes_fd = []
for i in bvfd:
    volumes_fd += list(get_box(all_volumes, all_centroids, i, False))
volumes_fd = rng.Range(volumes_fd)
# volumes_fd = get_box(all_volumes, all_centroids, bvfd, False)

volumes_f = rng.unite(volumes_fn, volumes_fd)
np.save('volumes_f', np.array(volumes_f))

D1_tag = mb.tag_get_handle('d1')
D2_tag = mb.tag_get_handle('d2')

finos = list(rng.unite(rng.unite(volumes_d, volumes_n), volumes_f))

vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices = rng.unite(vertices,mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids = np.array([mtu.get_average_position([v]) for v in vertices])

# volumes intermediarios por neumann
volumes_in = []
for i in bvin:
    volumes_in += list(get_box(vertices, all_vertex_centroids, i, False))
volumes_in = rng.Range(volumes_in)
# volumes_in = get_box(vertices, all_vertex_centroids, bvin, False)


# volumes intermediarios por Dirichlet
volumes_id = []
for i in bvid:
    volumes_id += list(get_box(vertices, all_vertex_centroids, i, False))
volumes_id = rng.Range(volumes_id)
# volumes_id = get_box(vertices, all_vertex_centroids, bvid, False)

intermediarios=rng.unite(volumes_id,volumes_in)
meshset_intermediarios = mb.create_meshset()
mb.add_entities(meshset_intermediarios, intermediarios)
intermediarios_tag = mb.tag_get_handle('intermediarios', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
mb.tag_set_data(intermediarios_tag, 0, meshset_intermediarios)
tags_criadas_aqui.append('intermediarios')

L1_ID_tag=mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L2_ID_tag=mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L3_ID_tag=mb.tag_get_handle("NIVEL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
tags_criadas_aqui += ['l1_ID', 'l2_ID', 'NIVEL_ID']

internos=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
ni = len(internos)
nf = len(faces)
na = len(arestas)
nv = len(vertices)
wirebasket_numbers =[ni, nf, na, nv]

tag_dual_1_meshset = mb.tag_get_handle('DUAL_1_MESHSET')
dual_1_meshset = mb.tag_get_data(tag_dual_1_meshset, 0, flat=True)[0]
meshsets_duais=mb.get_child_meshsets(dual_1_meshset)
k_eq_tag = mb.tag_get_handle("K_EQ")

t0 = time.time()
invbAii=solve_block_matrix(intern_adjs_by_dual, 0, mb, k_eq_tag, ni)
t1 = time.time()
print("inversão de Aii: ", t1-t0)

t0 = time.time()
invbAff = solve_block_matrix(faces_adjs_by_dual, ni, mb, k_eq_tag, nf)
t1 = time.time()
print("inversão de Aff: ", t1-t0)

# try:
#     SOL_ADM_f = np.load('SOL_ADM_fina.npy')
#     if len(SOL_ADM_f)!=len(all_volumes):
#         print("criará o vetor para refinamento")
#         SOL_ADM_f = np.repeat(1,len(all_volumes))
#     else:
#         print("leu o vetor criado")
# except:
#     print("criará o vetor para refinamento")
#     SOL_ADM_f = np.repeat(1,len(all_volumes))

ID_reordenado_tag = mb.tag_get_handle('ID_reord_tag')

# gids_p_d = mb.tag_get_data(ID_reordenado_tag,volumes_d,flat=True)
# gids_p_n = mb.tag_get_data(ID_reordenado_tag,volumes_n,flat=True)
# press_d=SOL_ADM_f[gids_p_d]
# press_n=SOL_ADM_f[gids_p_n]
# delta_p_res=abs(sum(press_d)/len(press_d)-sum(press_n)/len(press_n))
# delta_p_res=abs(max(press_d)-min(press_d))

l2_meshset_tag = mb.tag_get_handle('L2_MESHSET')
L2_meshset = mb.tag_get_data(l2_meshset_tag, 0, flat=True)[0]
med_perm_by_primal_1=[]
meshset_by_L2 = mb.get_child_meshsets(L2_meshset)
perm_tag = mb.tag_get_handle('PERM')

for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = mb.get_entities_by_handle(m1)
        perm1=mb.tag_get_data(perm_tag,elem_by_L1).reshape([len(elem_by_L1),9])
        med1_x=sum(perm1[:,0])/len(perm1[:,0])
        med_perm_by_primal_1.append(med1_x)
med_perm_by_primal_1=np.sort(med_perm_by_primal_1)

s=1.0    #Parâmetro da secante
lg=np.log(med_perm_by_primal_1)
ordem=11
fit=np.polyfit(range(len(lg)),lg,ordem)
x=sympy.Symbol('x',real=True,positive=True)
func=0
for i in range(ordem+1):
    func+=fit[i]*x**(ordem-i)
derivada=sympy.diff(func,x)
inc_secante=(lg[-1]-lg[0])/len(lg)

###########################################

equa=sympy.Eq(derivada,2*inc_secante)
#real_roots=sympy.solve(equa)
ind_inferior=int(sympy.nsolve(equa,0, verify=False))
if ind_inferior<0:
    ind_inferior=0
ind_superior=int(sympy.nsolve(equa,len(lg)))
if ind_superior>len(lg)-1:
    ind_superior=len(lg)-1

new_inc_secante=(lg[ind_superior]-lg[ind_inferior])/(ind_superior-ind_inferior)
eq2=sympy.Eq(derivada,new_inc_secante)
new_ind_inferior=int(sympy.nsolve(eq2,ind_inferior, verify=False))
if new_ind_inferior<ind_inferior:
    new_ind_inferior=ind_inferior
new_ind_superior=int(sympy.nsolve(eq2,ind_superior, verify=False))
if new_ind_superior>ind_superior:
    new_ind_superior=ind_superior
##########################################


val_barreira=med_perm_by_primal_1[ind_inferior]
val_canal=med_perm_by_primal_1[ind_superior]


t0 = time.time()
n1=0
n2=0
aux=0
meshset_by_L2 = mb.get_child_meshsets(L2_meshset)
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = mb.get_entities_by_handle(m1)
        ver_1=mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
        ver_1=rng.unite(ver_1,mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
        if ver_1[0] in finos:
            aux=1
            tem_poço_no_vizinho=True
        else:
            perm1=mb.tag_get_data(perm_tag,elem_by_L1).reshape([len(elem_by_L1),9])
            med1_x=sum(perm1[:,0])/len(perm1[:,0])
            var=sum((x - med1_x)**2 for x in perm1[:,0])/len(perm1[:,0])
            desv_pad_x=sqrt(var)

            med1_y=sum(perm1[:,4])/len(perm1[:,4])
            var=sum((x - med1_y)**2 for x in perm1[:,4])/len(perm1[:,4])
            desv_pad_y=sqrt(var)

            med1_z=sum(perm1[:,8])/len(perm1[:,8])
            var=sum((x - med1_z)**2 for x in perm1[:,8])/len(perm1[:,8])
            desv_pad_z=sqrt(var)

            ref=max([desv_pad_x/med1_x,desv_pad_y/med1_y,desv_pad_z/med1_z])
            desv_pad=max([desv_pad_x, desv_pad_y,desv_pad_z])
            #print("Desvio padrão",desv_pad_x,desv_pad_y,desv_pad_z,desv_pad_x/med1_x,desv_pad_y/med1_y,desv_pad_z/med1_z)
            #print(med1_z)
            gids_primal=mb.tag_get_data(ID_reordenado_tag,elem_by_L1)
            #print(delta_p_prim,delta_p_res)
            if med1_x<val_barreira or med1_x>val_canal:  #or max(press_primal)>max(press_n) or min(press_primal)<min(press_d)
                aux=1
                tem_poço_no_vizinho=True

        if ver_1[0] in intermediarios:
            tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1
                mb.tag_set_data(L1_ID_tag, elem, n1)
                mb.tag_set_data(L2_ID_tag, elem, n2)
                mb.tag_set_data(L3_ID_tag, elem, 1)
                finos.append(elem)
    if tem_poço_no_vizinho==False:
        elem_by_L2 = mb.get_entities_by_handle(m2)
        perm2=mb.tag_get_data(perm_tag,elem_by_L2).reshape([len(elem_by_L2),9])

        med2_x=sum(perm2[:,0])/len(perm2[:,0])
        var=sum((x - med2_x)**2 for x in perm2[:,0])/len(perm2[:,0])
        desv_pad_x2=sqrt(var)

        med2_y=sum(perm2[:,4])/len(perm2[:,4])
        var=sum((x - med2_y)**2 for x in perm2[:,4])/len(perm2[:,4])
        desv_pad_y2=sqrt(var)

        med2_z=sum(perm2[:,8])/len(perm2[:,8])
        var=sum((x - med2_z)**2 for x in perm2[:,8])/len(perm2[:,8])
        desv_pad_z2=sqrt(var)

        ref=max([desv_pad_x/med2_x,desv_pad_y/med2_y,desv_pad_z/med2_z])
        desv_pad=max([desv_pad_x2, desv_pad_y2,desv_pad_z2])


        desv_pad=sqrt(var)
        #print("Desvio padrão",desv_pad)
        if desv_pad>99991000 or ref>99993.6:
            tem_poço_no_vizinho=True
    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            ver_1=mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
            ver_1=rng.unite(ver_1,mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
            if ver_1[0] not in finos:
                mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(2,len(elem_by_L1)))
                t=0
            n1-=t
            n2-=t
    else:
        n2+=1
        for m1 in meshset_by_L1:
            elem_by_L1 = mb.get_entities_by_handle(m1)
            n1+=1
            mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
            mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
            mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(3,len(elem_by_L1)))

# ------------------------------------------------------------------------------
print('Definição da malha ADM: ',time.time()-t0)
t0=time.time()


av=mb.create_meshset()
mb.add_entities(av,all_volumes)
#mb.write_file('teste_3D_unstructured_18.vtk',[av])
#print("new file!!")


# fazendo os ids comecarem de 0 em todos os niveis

tags = [L1_ID_tag, L2_ID_tag]
for tag in tags:
    all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
    minim = min(all_gids)
    all_gids -= minim
    mb.tag_set_data(tag, all_volumes, all_gids)

###preprocessamento1
##########################################################

boundary_faces_tag = mb.tag_get_handle("FACES_BOUNDARY")
faces_boundary = mb.tag_get_data(boundary_faces_tag, 0, flat=True)[0]
faces_boundary = mb.get_entities_by_handle(faces_boundary)
faces_in = rng.subtract(all_faces, faces_boundary)

all_keqs = mb.tag_get_data(k_eq_tag, faces_in, flat=True)
Adjs = np.array([np.array(mb.get_adjacencies(face, 3)) for face in faces_in])

all_ids_reord = mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
map_volumes = dict(zip(all_volumes, range(len(all_volumes))))

tempo0_ADM=time.time()

lines_tf = []
cols_tf = []
data_tf = []

b2 = np.zeros(len(all_volumes))
s_grav_faces = np.zeros(len(faces_in))

for i, face in enumerate(faces_in):
    elem0 = Adjs[i][0]
    elem1 = Adjs[i][1]
    id0 = map_volumes[elem0]
    id1 = map_volumes[elem1]
    id_glob0 = all_ids_reord[id0]
    id_glob1 = all_ids_reord[id1]
    keq = all_keqs[i]

    lines_tf += [id_glob0, id_glob1]
    cols_tf += [id_glob1, id_glob0]
    data_tf += [keq, keq]

    s_grav = gama*keq*(all_centroids[id1][2] - all_centroids[id0][2])
    s_grav_faces[i] = s_grav
    s_grav *= -1
    b2[id_glob0] += s_grav
    b2[id_glob1] -= s_grav

if gravity:
    pass
else:
    b2 = np.zeros(len(all_volumes))

n = len(all_volumes)
Tf = csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))
Tf = Tf.tolil()
d1 = np.array(Tf.sum(axis=1)).reshape(1, n)[0]*(-1)
Tf.setdiag(d1)

scipy.sparse.save_npz('Tf', Tf.tocsc())
np.save('b2', b2)

As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, wirebasket_numbers)

ta1=time.time()

arestas_meshset=mb.create_meshset()
mb.add_entities(arestas_meshset,arestas)
faces_meshset=mb.create_meshset()
mb.add_entities(faces_meshset,faces)
internos_meshset=mb.create_meshset()
mb.add_entities(internos_meshset,internos)

nivel_0_arestas=mb.get_entities_by_type_and_tag(arestas_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_0_faces=mb.get_entities_by_type_and_tag(faces_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_0_internos=mb.get_entities_by_type_and_tag(internos_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))

IDs_arestas_0=mb.tag_get_data(ID_reordenado_tag,nivel_0_arestas,flat=True)
IDs_faces_0=mb.tag_get_data(ID_reordenado_tag,nivel_0_faces,flat=True)
IDs_internos_0=mb.tag_get_data(ID_reordenado_tag,nivel_0_internos,flat=True)

IDs_arestas_0_locais=np.subtract(IDs_arestas_0,ni+nf)
IDs_faces_0_locais=np.subtract(IDs_faces_0,ni)
IDs_internos_0_locais=IDs_internos_0

IDs_arestas_1_locais=np.setdiff1d(range(na),IDs_arestas_0_locais)

ids_arestas_slin_m0=np.nonzero(As['Aev'].sum(axis=1))[0]

Aev = As['Aev']
Ivv = As['Ivv']
Aif = As['Aif']
Afe = As['Afe']
invAee=lu_inv4(As['Aee'].tocsc(),ids_arestas_slin_m0)
M2=-invAee*Aev
PAD=vstack([M2,Ivv])

invAff=invbAff
M3=-invAff*(Afe*M2)

PAD=vstack([M3,PAD])
invAii=invbAii
PAD=vstack([-invAii*(Aif*M3),PAD])
print("get_OP_AMS", time.time()-ta1)

scipy.sparse.save_npz('PAD', PAD.tocsc())

del M3

ids_1=mb.tag_get_data(L1_ID_tag,vertices,flat=True)
fine_to_primal1_classic_tag = mb.tag_get_handle('FINE_TO_PRIMAL1_CLASSIC')
ids_class=mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
t0=time.time()

AMS_TO_ADM=dict(zip(ids_class,ids_1))
ty=time.time()
vm=mb.create_meshset()
mb.add_entities(vm,vertices)

tm=time.time()
PAD=csc_matrix(PAD)
OP1=PAD.copy()
OP3=PAD.copy()
nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
tor=time.time()
ID_global1=mb.tag_get_data(ID_reordenado_tag,nivel_0, flat=True)
IDs_ADM1=mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
#OP1[ID_global1]=csc_matrix((1,OP1.shape[1]))
IDs_AMS1=mb.tag_get_data(fine_to_primal1_classic_tag,nivel_0, flat=True)
OP3[ID_global1]=csc_matrix((1,OP3.shape[1]))

IDs_ADM1=mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)

IDs_ADM_1=mb.tag_get_data(L1_ID_tag,vertices, flat=True)
IDs_AMS_1=mb.tag_get_data(fine_to_primal1_classic_tag,vertices, flat=True)
lp=IDs_AMS_1
cp=IDs_ADM_1
dp=np.repeat(1,len(lp))

permut=csc_matrix((dp,(lp,cp)),shape=(len(vertices),n1))
opad3=OP3*permut
m=find(opad3)
l1=m[0]
c1=m[1]
d1=m[2]
l1=np.concatenate([l1,ID_global1])
c1=np.concatenate([c1,IDs_ADM1])
d1=np.concatenate([d1,np.ones(len(nivel_0))])
#opad3[ID_global1,IDs_ADM1]=np.ones(len(nivel_0))

opad3=csc_matrix((d1,(l1,c1)),shape=(len(all_volumes),n1))

print("opad1",tor-time.time(),time.time()-ta1, time.time()-tempo0_ADM)
OP_ADM=csc_matrix(opad3)

print("obteve OP_ADM_1",time.time()-tempo0_ADM)

l1=mb.tag_get_data(L1_ID_tag, all_volumes, flat=True)
c1=mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
d1=np.ones((1,len(l1)),dtype=np.int)[0]
OR_ADM=csc_matrix((d1,(l1,c1)),shape=(n1,len(all_volumes)))

l1=mb.tag_get_data(fine_to_primal1_classic_tag, all_volumes, flat=True)
c1=mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
d1=np.ones((1,len(l1)),dtype=np.int)[0]
OR_AMS=csc_matrix((d1,(l1,c1)),shape=(nv,len(all_volumes)))

scipy.sparse.save_npz('OR_AMS', OR_AMS.tocsc())

OP_AMS=PAD

v=mb.create_meshset()
mb.add_entities(v,vertices)

inte=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

fine_to_primal2_classic_tag = mb.tag_get_handle('FINE_TO_PRIMAL2_CLASSIC')

lines=[]
cols=[]
data=[]

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
tu=time.time()
for i in range(nint):
    v=inte[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(i)
    cols.append(ID_AMS)
    data.append(1)

    #G[i][ID_AMS]=1
i=0
for i in range(nfac):
    v=fac[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+i][ID_AMS]=1
i=0
for i in range(nare):
    v=are[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+nfac+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+nfac+i][ID_AMS]=1
i=0
for i in range(nver):
    v=ver[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+nfac+nare+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+nfac+nare+i][ID_AMS]=1
G=csc_matrix((data,(lines,cols)),shape=(nv,nv))

scipy.sparse.save_npz('G', G.tocsc())

T = Tf.copy()
b = b2.copy()

T_AMS=OR_AMS*T*OP_AMS
W_AMS=G*T_AMS*G.transpose()

MPFA_NO_NIVEL_2=data_loaded['MPFA']

nv1=nv

ni=nint
nf=nfac
na=nare
nv=nver
wirebasket_numbers_nv2 = [ni, nf, na, nv]

Aii=W_AMS[0:ni,0:ni]
Aif=W_AMS[0:ni,ni:ni+nf]
Aie=W_AMS[0:ni,ni+nf:ni+nf+na]
Aiv=W_AMS[0:ni,ni+nf+na:ni+nf+na+nv]

lines=[]
cols=[]
data=[]
if MPFA_NO_NIVEL_2 ==False:
    for i in range(ni):
        lines.append(i)
        cols.append(i)
        data.append(float(Aie.sum(axis=1)[i])+float(Aiv.sum(axis=1)[i]))
    S=csc_matrix((data,(lines,cols)),shape=(ni,ni))
    Aii += S
    del(S)

Afi=W_AMS[ni:ni+nf,0:ni]
Aff=W_AMS[ni:ni+nf,ni:ni+nf]
Afe=W_AMS[ni:ni+nf,ni+nf:ni+nf+na]
Afv=W_AMS[ni:ni+nf,ni+nf+na:ni+nf+na+nv]

lines=[]
cols=[]
data_fi=[]
data_fv=[]
for i in range(nf):
    lines.append(i)
    cols.append(i)
    data_fi.append(float(Afi.sum(axis=1)[i]))
    data_fv.append(float(Afv.sum(axis=1)[i]))

Sfi=csc_matrix((data_fi,(lines,cols)),shape=(nf,nf))
Aff += Sfi
if MPFA_NO_NIVEL_2==False:
    Sfv=csc_matrix((data_fv,(lines,cols)),shape=(nf,nf))
    Aff +=Sfv

Aei=W_AMS[ni+nf:ni+nf+na,0:ni]
Aef=W_AMS[ni+nf:ni+nf+na,ni:ni+nf]
Aee=W_AMS[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
Aev=W_AMS[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]

Avv=W_AMS[ni+nf+na:ni+nf+na+nv,ni+nf+na:ni+nf+na+nv]

lines=[]
cols=[]
data=[]
for i in range(na):
    lines.append(i)
    cols.append(i)
    data.append(float(Aei.sum(axis=1)[i])+float(Aef.sum(axis=1)[i]))
S=csc_matrix((data,(lines,cols)),shape=(na,na))
Aee += S

Ivv = scipy.sparse.identity(nv)
invAee = lu_inv2(Aee)
M2 = -csc_matrix(invAee)*Aev
P2 = vstack([M2, Ivv])

invAff = lu_inv2(Aff)

if MPFA_NO_NIVEL_2:
    M3 = -invAff*Afe*M2-invAff*Afv
    P2 = vstack([M3, P2])
else:
    Mf = -invAff*Afe*M2
    P2 = vstack([Mf, P2])
invAii = lu_inv2(Aii)
if MPFA_NO_NIVEL_2:
    M3 = invAii*(-Aif*M3+Aie*invAee*Aev-Aiv)
    P2 = vstack([M3, P2])
else:
    P2 = vstack([-invAii*Aif*Mf, P2])

COL_TO_ADM_2 = {}
# ver é o meshset dos vértices da malha dual grossa
for i in range(nv):
    v=ver[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal2_classic_tag,v))
    ID_ADM=int(mb.tag_get_data(L2_ID_tag,v))
    COL_TO_ADM_2[str(i)] = ID_ADM

P2=G.transpose()*P2

OP_AMS_2=P2

scipy.sparse.save_npz('OP_AMS_2', OP_AMS_2.tocsc())



ID_AMS_1=mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
ID_AMS_2=mb.tag_get_data(fine_to_primal2_classic_tag,vertices,flat=True)

OR_AMS_2=csc_matrix((np.repeat(1,len(vertices)),(ID_AMS_2,ID_AMS_1)),shape=(len(ver),len(vertices)))
scipy.sparse.save_npz('OR_AMS_2', OR_AMS_2.tocsc())
T_AMS_2=OR_AMS_2*T_AMS*OP_AMS_2

lines=[]
cols=[]
data=[]

m_vert=mb.create_meshset()
mb.add_entities(m_vert,vertices)
nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=mb.get_entities_by_type_and_tag(m_vert, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))

P2=csc_matrix(P2)

matriz=scipy.sparse.find(P2)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
IDs_ADM_1=mb.tag_get_data(L1_ID_tag,nivel_1,flat=True)
IDs_ADM_2=mb.tag_get_data(L2_ID_tag,nivel_1,flat=True)
IDs_AMS=mb.tag_get_data(fine_to_primal1_classic_tag, nivel_1,flat=True)
dd=np.array([])
for i in range(len(nivel_1)):
    ID_AMS = IDs_AMS[i]
    dd=np.concatenate([dd,np.array(np.where(LIN==ID_AMS))[0]])

    #print(ID_ADM_1,ID_ADM_2,ID_AMS,len(dd[0]),"-------")
LIN=np.delete(LIN,dd,axis=0)
COL=np.delete(COL,dd,axis=0)
DAT=np.delete(DAT,dd,axis=0)

lines=IDs_ADM_1
cols=IDs_ADM_2
data=np.ones((1,len(lines)),dtype=np.int32)[0]

IDs_ADM_1=mb.tag_get_data(L1_ID_tag,nivel_0,flat=True)
IDs_ADM_2=mb.tag_get_data(L2_ID_tag,nivel_0,flat=True)
IDs_AMS=mb.tag_get_data(fine_to_primal1_classic_tag, nivel_0,flat=True)
tu=time.time()
dd=np.array([])
for i in range(len(nivel_0)):
    ID_AMS = IDs_AMS[i]
    dd=np.concatenate([dd,np.array(np.where(LIN==ID_AMS))[0]])
    dd=np.unique(dd)
    #dd=np.where(LIN==ID_AMS)
    #LIN=np.delete(LIN,dd,axis=0)
    #COL=np.delete(COL,dd,axis=0)
    #DAT=np.delete(DAT,dd,axis=0)

LIN=np.delete(LIN,dd,axis=0)
COL=np.delete(COL,dd,axis=0)
DAT=np.delete(DAT,dd,axis=0)
print("fss",time.time()-tu)
    #print(ID_AMS,ID_ADM_1,ID_ADM_2,dd)

lines=np.concatenate([lines,IDs_ADM_1])
cols=np.concatenate([cols,IDs_ADM_2])
data=np.concatenate([data,np.ones((1,len(IDs_ADM_1)),dtype=np.int32)[0]])


LIN_ADM=[AMS_TO_ADM[k] for k in LIN]
COL_ADM=[COL_TO_ADM_2[str(k)] for k in COL]
lines=np.concatenate([lines,LIN_ADM])
cols=np.concatenate([cols,COL_ADM])
data=np.concatenate([data,DAT])
#
#del(COL)
#del(LIN)
#del(DAT)

OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1,n2))

# OP2=P2.copy()
# ta2=time.time()
# vm=mb.create_meshset()
# mb.add_entities(vm,vertices)
# v_nivel_0=mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
# v_nivel_1=mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
# v_n0e1=rng.unite(v_nivel_0,v_nivel_1)
# ID_class=mb.tag_get_data(fine_to_primal1_classic_tag,v_n0e1, flat=True)
# OP2[ID_class]=csc_matrix((1,OP2.shape[1]))

nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
n0e1=rng.unite(nivel_0,nivel_1)
IDs_ADM1=mb.tag_get_data(L1_ID_tag,n0e1, flat=True)
IDs_ADM2=mb.tag_get_data(L2_ID_tag,n0e1, flat=True)

# lines=IDs_ADM1
# cols=IDs_ADM2
# data=np.repeat(1,len(lines))
# IDs_AMS2=mb.tag_get_data(fine_to_primal2_classic_tag,ver, flat=True)
# IDs_ADM2_ver=mb.tag_get_data(L2_ID_tag,ver, flat=True)
# AMS2_TO_ADM2=dict(zip(IDs_AMS2,IDs_ADM2_ver))

# m=find(OP2)
# l1=m[0]
# c1=m[1]
# d1=m[2]
# ID_ADM2=[AMS2_TO_ADM2[k] for k in c1]
# lines=np.concatenate([lines,l1])
# cols=np.concatenate([cols,ID_ADM2])
# data=np.concatenate([data,d1])
#
# opad2=csc_matrix((data,(lines,cols)),shape=(n1,n2))
# print("opad2",time.time()-ta2)

l2=mb.tag_get_data(L2_ID_tag, all_volumes, flat=True)
c2=mb.tag_get_data(L1_ID_tag, all_volumes, flat=True)
d2=np.ones((1,len(l2)),dtype=np.int)[0]
OR_ADM_2=csc_matrix((d2,(l2,c2)),shape=(n2,n1))

dirichlet_tag = mb.tag_get_handle('P')
ID_global=mb.tag_get_data(ID_reordenado_tag,volumes_d, flat=True)
values_d = mb.tag_get_data(dirichlet_tag, volumes_d, flat=True)
T[ID_global]=scipy.sparse.lil_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))
b[ID_global] = values_d

neuman_tag = mb.tag_get_handle('Q')
ID_globaln=mb.tag_get_data(ID_reordenado_tag,volumes_n, flat=True)
values_n = mb.tag_get_data(neuman_tag, volumes_n, flat=True)
if len(ID_globaln) > 0:
    b[ID_globaln] = values_n

scipy.sparse.save_npz('T', T.tocsc())
np.save('b', b)

t0=time.time()
SOL_ADM_2=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b) #+OR_ADM_2*T1*corr_adm2_sd    -OR_ADM_2*T1*corr_adm2_sd

SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM_2#+OP_ADM*corr_adm2_sd#.transpose().toarray()[0] #+corr_adm1_sd.transpose().toarray()[0]
print("Solução do sistema ADM",time.time()-t0)
np.save('SOL_ADM_fina.npy', SOL_ADM_fina)
#teste=OP_ADM*(-OP_ADM_2*OR_ADM_2*T1*corr_adm2_sd+corr_adm2_sd)

print("TEMPO TOTAL PARA SOLUÇÃO ADM:", time.time()-tempo0_ADM)
print("")
#mb.write_file('teste_3D_unstructured_18_2.vtk',[av])
Sol_ADM_tag=mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

perm_xx_tag=mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
GIDs=mb.tag_get_data(ID_reordenado_tag,all_volumes,flat=True)
perms_xx=mb.tag_get_data(perm_tag,all_volumes)[:,0]
cont=0
for v in all_volumes:
    gid=GIDs[cont]
    mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])
    mb.tag_set_data(perm_xx_tag,v,perms_xx[cont])
    cont+=1
av=mb.create_meshset()
mb.add_entities(av,all_volumes)


SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)    #-OR_ADM*T*corr_adm1_sd   +OR_ADM*T*corr_adm1_sd



SOL_ADM_fina_1=OP_ADM*SOL_ADM_1#-corr_adm1_sd.transpose()[0]
'''
SOL_TPFA = np.load('SOL_TPFA.npy')'''

print("resolvendo TPFA")
t0=time.time()
T = T.tocsc()
SOL_TPFA=linalg.spsolve(T,b)
print("resolveu TPFA: ")
np.save('SOL_TPFA.npy', SOL_TPFA)


erro=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erro[i]=100*abs((SOL_TPFA[i]-SOL_ADM_fina[i])/SOL_TPFA[i])

erroADM1=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)): erroADM1[i]=100*abs((SOL_TPFA[i]-SOL_ADM_fina_1[i])/SOL_TPFA[i])

ERRO_tag=mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROadm1_tag=mb.tag_get_handle("erroADM1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag=mb.tag_get_handle("Pressão TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag=mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

erm_xx_tag=mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
GIDs=mb.tag_get_data(ID_reordenado_tag,all_volumes,flat=True)
perms_xx=mb.tag_get_data(perm_tag,all_volumes)[:,0]
cont=0
for v in all_volumes:
    gid=GIDs[cont]
    mb.tag_set_data(ERRO_tag,v,erro[gid])
    mb.tag_set_data(ERROadm1_tag,v,erroADM1[gid])
    mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])
    mb.tag_set_data(perm_xx_tag,v,perms_xx[cont])
    cont+=1

input_file = data_loaded['input_file']
ext_vtk = input_file + '_malha_adm.vtk'
ext_h5m_out = input_file + '_malha_adm.h5m'

mb.write_file(ext_vtk,[av])
mb.write_file(ext_h5m_out)
print('New file created')

print('saiu mad_mesh')
