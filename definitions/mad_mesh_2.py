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
ext_h5m = input_file + '_malha_adm.h5m'
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
# wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR')
# wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER')
# wells_injector = mb.tag_get_data(wells_injector_tag, 0, flat=True)[0]
# wells_producer = mb.tag_get_data(wells_producer_tag, 0, flat=True)[0]
# wells_injector = mb.get_entities_by_handle(wells_injector)
# wells_producer = mb.get_entities_by_handle(wells_producer)
#
#
# verts = mb.get_connectivity(all_volumes)
# coords = mb.get_coords(verts).reshape([len(verts), 3])
# mins = coords.min(axis=0)
# maxs = coords.max(axis=0)
# Ltotal = maxs - mins + 1e-9  # correcao de ponto flutuante
# Lx, Ly, Lz = Ltotal

# bvd = []
# bvn = []
#
# inds_wells = []
# for well in data_loaded['Wells_structured']:
#     w = data_loaded['Wells_structured'][well]
#     if w['type_region'] == 'box':
#         box_volumes = np.array([np.array(w['region1']), np.array(w['region2'])])
#         inds0 = np.where(all_centroids[:,0] > box_volumes[0,0])[0]
#         inds1 = np.where(all_centroids[:,1] > box_volumes[0,1])[0]
#         inds2 = np.where(all_centroids[:,2] > box_volumes[0,2])[0]
#         c1 = set(inds0) & set(inds1) & set(inds2)
#         inds0 = np.where(all_centroids[:,0] < box_volumes[1,0])[0]
#         inds1 = np.where(all_centroids[:,1] < box_volumes[1,1])[0]
#         inds2 = np.where(all_centroids[:,2] < box_volumes[1,2])[0]
#         c2 = set(inds0) & set(inds1) & set(inds2)
#         inds_vols = list(c1 & c2)
#         inds_wells += inds_vols
#         volumes = rng.Range(np.array(all_volumes)[inds_vols])
#     else:
#         raise NameError("Defina o tipo de regiao em type_region: 'box'")
#
#     value = float(w['value'])
#
#     if w['type_prescription'] == 'dirichlet':
#         bvd.append(box_volumes)
#         if gravity == False:
#             pressao = np.repeat(value, len(volumes))
#
#         else:
#             z_elems_d = -1*mb.tag_get_data(cent_tag, volumes)[:,2]
#             delta_z = z_elems_d + Lz
#             pressao = gama*(delta_z) + value
#
#         mb.tag_set_data(press_value_tag, volumes, pressao)
#
#     elif  w['type_prescription'] == 'neumann':
#         bvn.append(box_volumes)
#         value = value/len(volumes)
#         if w['type_well'] == 'injector':
#             value = -value
#         mb.tag_set_data(vazao_value_tag, volumes, np.repeat(value, len(volumes)))
#
#     else:
#         raise NameError("type_prescription == 'neumann' or 'dirichlet'")
#
#     if w['type_well'] == 'injector':
#         mb.add_entities(wells_injector, volumes)
#     else:
#         mb.add_entities(wells_producer, volumes)
#
#
# bvd = bvd[0]
# bvn = bvn[0]
#
# bvfd = np.array([np.array([bvd[0][0]-r0, bvd[0][1]-r0, bvd[0][2]-r0]), np.array([bvd[1][0]+r0, bvd[1][1]+r0, bvd[1][2]+r0])])
# bvfn = np.array([np.array([bvn[0][0]-r0, bvn[0][1]-r0, bvn[0][2]-r0]), np.array([bvn[1][0]+r0, bvn[1][1]+r0, bvn[1][2]+r0])])
#
# bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
# bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])
#
# volumes_d, inds_vols_d= get_box(all_volumes, all_centroids, bvd, True)
#
# # volumes com vazao prescrita
# volumes_n, inds_vols_n = get_box(all_volumes, all_centroids, bvn, True)
#
# # volumes finos por neumann
# volumes_fn = get_box(all_volumes, all_centroids, bvfn, False)
#
# # volumes finos por Dirichlet
# volumes_fd = get_box(all_volumes, all_centroids, bvfd, False)
#
# volumes_f = rng.unite(volumes_fn, volumes_fd)
#
# inds_pocos = inds_vols_d + inds_vols_n
# Cent_wels = all_centroids[inds_pocos]
cent_tag = mb.tag_get_handle('CENT')
D1_tag = mb.tag_get_handle('d1')
D2_tag = mb.tag_get_handle('d2')
dirichlet_tag = mb.tag_get_handle('P')
neumann_tag = mb.tag_get_handle('Q')
intermediarios_tag = mb.tag_get_handle('intermediarios')
intermediarios = mb.tag_get_data(intermediarios_tag, 0, flat=True)[0]
intermediarios = mb.get_entities_by_handle(intermediarios)

volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dirichlet_tag]), np.array([None]))
volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([neumann_tag]), np.array([None]))
volumes_f = rng.Range(np.load('volumes_f.npy'))

finos = list(rng.unite(rng.unite(volumes_d, volumes_n), volumes_f))

vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices = rng.unite(vertices,mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids = mb.tag_get_data(cent_tag, vertices)

L1_ID_tag=mb.tag_get_handle("l1_ID")
L2_ID_tag=mb.tag_get_handle("l2_ID")
L3_ID_tag=mb.tag_get_handle("NIVEL_ID")

internos=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
ni = len(internos)
nf = len(faces)
na = len(arestas)
nv = len(vertices)
wirebasket_numbers =[ni, nf, na, nv]

SOL_ADM_f = np.load('SOL_ADM_fina.npy')

ID_reordenado_tag = mb.tag_get_handle('ID_reord_tag')

pocos = rng.unite(volumes_d, volumes_n)
gids_pocos = mb.tag_get_data(ID_reordenado_tag, pocos, flat=True)
press_d=SOL_ADM_f[gids_pocos]

# delta_p_res=abs(sum(press_d)/len(press_d)-sum(press_n)/len(press_n))
delta_p_res=abs(max(press_d)-min(press_d))

l2_meshset_tag = mb.tag_get_handle('L2_MESHSET')
L2_meshset = mb.tag_get_data(l2_meshset_tag, 0, flat=True)[0]
med_perm_by_primal_1=[]
meshset_by_L2 = mb.get_child_meshsets(L2_meshset)
perm_tag = mb.tag_get_handle('PERM')

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

            gids_primal=mb.tag_get_data(ID_reordenado_tag,elem_by_L1)
            press_primal=SOL_ADM_f[gids_primal]
            delta_p_prim=max(press_primal)-min(press_primal)
            #print(delta_p_prim,delta_p_res)
            if delta_p_prim>0.2*delta_p_res:  #or max(press_primal)>max(press_n) or min(press_primal)<min(press_d)
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
        elem_by_L2 = mb.get_entities_by_handle(m2)
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


boundary_faces_tag = mb.tag_get_handle("FACES_BOUNDARY")
faces_boundary = mb.tag_get_data(boundary_faces_tag, 0, flat=True)[0]
faces_boundary = mb.get_entities_by_handle(faces_boundary)
faces_in = rng.subtract(all_faces, faces_boundary)

all_ids_reord = mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
map_volumes = dict(zip(all_volumes, range(len(all_volumes))))

tempo0_ADM=time.time()

Tf = scipy.sparse.load_npz('Tf.npz')
b2 = np.load('b2.npy')

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


PAD = scipy.sparse.load_npz('PAD.npz')


ids_1=mb.tag_get_data(L1_ID_tag,vertices,flat=True)
fine_to_primal1_classic_tag = mb.tag_get_handle('FINE_TO_PRIMAL1_CLASSIC')
ids_class=mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
t0=time.time()

AMS_TO_ADM=dict(zip(ids_class,ids_1))
ty=time.time()
vm=mb.create_meshset()
mb.add_entities(vm,vertices)

tm=time.time()
PAD=PAD.tocsc()
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

OR_AMS=scipy.sparse.load_npz('OR_AMS.npz')

OP_AMS=PAD

v=mb.create_meshset()
mb.add_entities(v,vertices)

inte=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

fine_to_primal2_classic_tag = mb.tag_get_handle('FINE_TO_PRIMAL2_CLASSIC')

mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

lines=[]
cols=[]
data=[]

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
tu=time.time()

G=scipy.sparse.load_npz('G.npz')

T = Tf.copy()
b = b2.copy()

T_AMS=OR_AMS*T*OP_AMS

OP_AMS_2=scipy.sparse.load_npz('OP_AMS_2.npz')
P2 = OP_AMS_2



# ID_AMS_1=mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
# ID_AMS_2=mb.tag_get_data(fine_to_primal2_classic_tag,vertices,flat=True)

OR_AMS_2=scipy.sparse.load_npz('OR_AMS_2.npz')

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
#
OP2=P2.copy()
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

lines=IDs_ADM1
cols=IDs_ADM2
data=np.repeat(1,len(lines))
IDs_AMS2=mb.tag_get_data(fine_to_primal2_classic_tag,ver, flat=True)
IDs_ADM2_ver=mb.tag_get_data(L2_ID_tag,ver, flat=True)
AMS2_TO_ADM2=dict(zip(IDs_AMS2,IDs_ADM2_ver))

m=find(OP2)
l1=m[0]
c1=m[1]
d1=m[2]
ID_ADM2=[AMS2_TO_ADM2[k] for k in c1]
lines=np.concatenate([lines,l1])
cols=np.concatenate([cols,ID_ADM2])
data=np.concatenate([data,d1])

opad2=csc_matrix((data,(lines,cols)),shape=(n1,n2))
print("opad2")

l2=mb.tag_get_data(L2_ID_tag, all_volumes, flat=True)
c2=mb.tag_get_data(L1_ID_tag, all_volumes, flat=True)
d2=np.ones((1,len(l2)),dtype=np.int)[0]
OR_ADM_2=csc_matrix((d2,(l2,c2)),shape=(n2,n1))

dirichlet_tag = mb.tag_get_handle('P')
ID_global=mb.tag_get_data(ID_reordenado_tag,volumes_d, flat=True)
values_d = mb.tag_get_data(dirichlet_tag, volumes_d, flat=True)

T = scipy.sparse.load_npz('T.npz')
b = np.load('b.npy')

########################## fim de apagar

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
# SOL_TPFA=linalg.spsolve(T,b)
SOL_TPFA = np.load('SOL_TPFA.npy')
print("resolveu TPFA: ")


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
ext_vtk = input_file + '_malha_adm_final.vtk'
ext_h5m_output = input_file + '_malha_adm_final.h5m'

mb.write_file(ext_vtk,[av])
mb.write_file(ext_h5m_output)
print('New file created')

print('saiu mad_mesh_2')
