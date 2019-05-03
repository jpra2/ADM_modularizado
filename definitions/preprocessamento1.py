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
        if gravity == False:
            pressao = np.repeat(value, len(volumes))

        else:
            z_elems_d = -1*mb.tag_get_data(cent_tag, volumes)[:,2]
            delta_z = z_elems_d + Lz
            pressao = gama*(delta_z) + value

        mb.tag_set_data(press_value_tag, volumes, pressao)

    elif  w['type_prescription'] == 'neumann':
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


bvd = bvd[0]
bvn = bvn[0]

bvfd = np.array([np.array([bvd[0][0]-r0, bvd[0][1]-r0, bvd[0][2]-r0]), np.array([bvd[1][0]+r0, bvd[1][1]+r0, bvd[1][2]+r0])])
bvfn = np.array([np.array([bvn[0][0]-r0, bvn[0][1]-r0, bvn[0][2]-r0]), np.array([bvn[1][0]+r0, bvn[1][1]+r0, bvn[1][2]+r0])])

bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])

volumes_d, inds_vols_d= get_box(all_volumes, all_centroids, bvd, True)

# volumes com vazao prescrita
volumes_n, inds_vols_n = get_box(all_volumes, all_centroids, bvn, True)

# volumes finos por neumann
volumes_fn = get_box(all_volumes, all_centroids, bvfn, False)

# volumes finos por Dirichlet
volumes_fd = get_box(all_volumes, all_centroids, bvfd, False)

volumes_f = rng.unite(volumes_fn, volumes_fd)

inds_pocos = inds_vols_d + inds_vols_n
Cent_wels = all_centroids[inds_pocos]

D1_tag = mb.tag_get_handle('d1')
D2_tag = mb.tag_get_handle('d2')

finos = list(rng.unite(rng.unite(volumes_d, volumes_n), volumes_f))

vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices = rng.unite(vertices,mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids = np.array([mtu.get_average_position([v]) for v in vertices])

# volumes intermediarios por neumann
volumes_in = get_box(vertices, all_vertex_centroids, bvin, False)

# volumes intermediarios por Dirichlet
volumes_id = get_box(vertices, all_vertex_centroids, bvid, False)
intermediarios=rng.unite(volumes_id,volumes_in)

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

SOL_ADM_f = np.repeat(1,len(all_volumes))
print('criara vetor para refinamento')

ID_reordenado_tag = mb.tag_get_handle('ID_reord_tag')

gids_p_d = mb.tag_get_data(ID_reordenado_tag,volumes_d,flat=True)
gids_p_n = mb.tag_get_data(ID_reordenado_tag,volumes_n,flat=True)
press_d=SOL_ADM_f[gids_p_d]
press_n=SOL_ADM_f[gids_p_n]

delta_p_res=abs(sum(press_d)/len(press_d)-sum(press_n)/len(press_n))

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
            press_primal=SOL_ADM_f[gids_primal]
            delta_p_prim=max(press_primal)-min(press_primal)
            #print(delta_p_prim,delta_p_res)
            if delta_p_prim>3*delta_p_res or ref>5.0 or med1_x<val_barreira or med1_x>val_canal:  #or max(press_primal)>max(press_n) or min(press_primal)<min(press_d)
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

SOL_ADM_f = np.repeat(1,len(all_volumes))
ID_reordenado_tag = mb.tag_get_handle('ID_reord_tag')


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





av = mb.create_meshset()
mb.add_entities(av, all_volumes)

ext_h5m_out = input_file + 'prep1.h5m'
ext_vtk_out = input_file + 'prep1.vtk'
mb.write_file(ext_h5m_out)
mb.write_file(ext_vtk_out, [av])
