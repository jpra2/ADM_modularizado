import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import shutil
import yaml
# import pdb; pdb.set_trace()
from definitions.mesh_manager import MeshManager
from definitions.functions1 import *
from utils import pymoab_utils as utpy


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')

### deletar o conteudo das pastas no diretorio flying
shutil.rmtree(mono_dir)
shutil.rmtree(bif_dir)
shutil.rmtree(flying_dir)
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
bifasico = data_loaded['bifasico']
os.chdir(flying_dir)

def Min_Max(e):
    verts = MM.mb.get_connectivity(e)
    coords = MM.mb.get_coords(verts).reshape([len(verts),3])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return([mins[0],maxs[0],mins[1],maxs[1],mins[2],maxs[2]])

all_volumes=MM.all_volumes
all_centroids = MM.all_centroids
for i, j in zip(all_volumes, all_centroids):
    MM.mb.tag_set_data(MM.cent_tag, i, j)

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

tempo0_pre=time.time()

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/l2[0])):    lx2.append(mins[0]+i*l2[0])
for i in range(int(Ly/l2[1])):    ly2.append(mins[1]+i*l2[1])
for i in range(int(Lz/l2[2])):    lz2.append(mins[2]+i*l2[2])
lx2.append(Lx)
ly2.append(Ly)
lz2.append(Lz)

#-------------------------------------------------------------------------------
# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2[0]/l1[0])):   lx1.append(i*l1[0])
for i in range(int(l2[1]/l1[1])):   ly1.append(i*l1[1])
for i in range(int(l2[2]/l1[2])):   lz1.append(i*l1[2])

D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
nD_x=int((D_x+0.001)/l1[0])
nD_y=int((D_y+0.001)/l1[1])
nD_z=int((D_z+0.001)/l1[2])

lxd1=[mins[0]+dx0/100]
for i in range(int(Lx/l1[0])-2-nD_x):
    lxd1.append(l1[0]/2+(i+1)*l1[0])
lxd1.append(mins[0]+Lx-dx0/100)

lyd1=[mins[1]+dy0/100]
for i in range(int(Ly/l1[1])-2-nD_y):
    lyd1.append(l1[1]/2+(i+1)*l1[1])
lyd1.append(mins[1]+Ly-dy0/100)

lzd1=[mins[2]+dz0/100]
for i in range(int(Lz/l1[2])-2-nD_z):
    lzd1.append(l1[2]/2+(i+1)*l1[2])
lzd1.append(mins[2]+Lz-dz0/100)

print("definiu planos do nível 1")
lxd2=[lxd1[0]]
for i in range(1,int(len(lxd1)*l1[0]/l2[0])-1):
    lxd2.append(lxd1[int(i*l2[0]/l1[0] + 1e-9)+1])
lxd2.append(lxd1[-1])

lyd2=[lyd1[0]]
for i in range(1,int(len(lyd1)*l1[1]/l2[1])-1):
    lyd2.append(lyd1[int(i*l2[1]/l1[1]+1e-9)+1])
lyd2.append(lyd1[-1])

lzd2=[lzd1[0]]
for i in range(1,int(len(lzd1)*l1[2]/l2[2])-1):
    lzd2.append(lzd1[int(i*l2[2]/l1[2] + 1e-9)+1])
lzd2.append(lzd1[-1])

print("definiu planos do nível 2")


t0=time.time()
L2_meshset=MM.mb.create_meshset()       # root Meshset
l2_meshset_tag = MM.mb.tag_get_handle('L2_MESHSET', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
MM.mb.tag_set_data(l2_meshset_tag, 0, L2_meshset)
tags_criadas_aqui.append('L2_MESHSET')

D1_tag=MM.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
D2_tag=MM.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################
tags_criadas_aqui += ['d1', 'd2']

fine_to_primal1_classic_tag = MM.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
fine_to_primal2_classic_tag = MM.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
AV_meshset=MM.mb.create_meshset()
tags_criadas_aqui += ['FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC']

primal_id_tag1 = MM.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
primal_id_tag2 = MM.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
tags_criadas_aqui += ['PRIMAL_ID_1', 'PRIMAL_ID_2']

nc1=0
nc2=0

centroids=all_centroids
sx=0
ref_dual=False
MM.mb.add_entities(AV_meshset,all_volumes)
for i in range(len(lx2)-1):
    t1=time.time()
    if i==len(lx2)-2:
        sx=D_x
    sy=0
    for j in range(len(ly2)-1):
        if j==len(ly2)-2:
            sy=D_y
        sz=0
        for k in range(len(lz2)-1):
            if k==len(lz2)-2:
                sz=D_z
            l2_meshset=MM.mb.create_meshset()
            cont=0
            box_primal2 = np.array([np.array([lx2[i], ly2[j], lz2[k]]), np.array([lx2[i]+l2[0]+sx, ly2[j]+l2[1]+sy, lz2[k]+l2[2]+sz])])
            elem_por_L2 = get_box(MM.all_volumes, centroids, box_primal2, False)

            #if i<len(lxd2)-1 and j<len(lyd2)-1 and k<len(lzd2)-1:
            #    box_dual2 = np.array([np.array([lxd2[i], lyd2[j], lzd2[k]]), np.array([lxd2[i+1], lyd2[j+1], lzd2[k+1]])])
            #    elem_por_D2 = get_box(MM.all_volumes, centroids, box_dual2, False)

            MM.mb.add_entities(l2_meshset,elem_por_L2)
            centroid_p2=np.array([MM.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            for elem in elem_por_L2:
                centroid=centroid_p2[cont]
                cont+=1
                f1a2v3=0
                if (centroid[0]-lxd2[i])**2<=l1[0]**2/4:
                    f1a2v3+=1
                if (centroid[1]-lyd2[j])**2<=l1[1]**2/4:
                    f1a2v3+=1
                if (centroid[2]-lzd2[k])**2<=l1[2]**2/4:
                    f1a2v3+=1
                MM.mb.tag_set_data(D2_tag, elem, f1a2v3)
                MM.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)

            MM.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=MM.mb.get_entities_by_handle(l2_meshset)
            print(k, len(sg), time.time()-t1)
            t1=time.time()
            MM.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)

            centroids_primal2=np.array([MM.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            nc2+=1
            s1x=0
            for m in range(len(lx1)):
                a=int(l2[0]/l1[0])*i+m
                if Lx-D_x==lx2[i]+lx1[m]+l1[0]:# and D_x==Lx-int(Lx/l1[0])*l1[0]:
                    s1x=D_x
                s1y=0
                for n in range(len(ly1)):
                    b=int(l2[1]/l1[1])*j+n
                    if Ly-D_y==ly2[j]+ly1[n]+l1[1]:# and D_y==Ly-int(Ly/l1[1])*l1[1]:
                        s1y=D_y
                    s1z=0

                    for o in range(len(lz1)):
                        c=int(l2[2]/l1[2])*k+o
                        if Lz-D_z==lz2[k]+lz1[o]+l1[2]:
                            s1z=D_z
                        l1_meshset=MM.mb.create_meshset()

                        box_primal1 = np.array([np.array([lx2[i]+lx1[m], ly2[j]+ly1[n], lz2[k]+lz1[o]]), np.array([lx2[i]+lx1[m]+l1[0]+s1x, ly2[j]+ly1[n]+l1[1]+s1y, lz2[k]+lz1[o]+l1[2]+s1z])])
                        elem_por_L1 = get_box(elem_por_L2, centroids_primal2, box_primal1, False)
                        MM.mb.add_entities(l1_meshset,elem_por_L1)
                        #centroid_p1=np.array([MM.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L1])
                        cont1=0
                        values_1=[]
                        faces1=[]
                        internos1=[]
                        for e in elem_por_L1:
                            #centroid=centroid_p1[cont1]
                            cont1+=1
                            f1a2v3=0
                            M_M=Min_Max(e)
                            if (M_M[0]<lxd1[a] and M_M[1]>=lxd1[a]):
                                f1a2v3+=1
                            if (M_M[2]<lyd1[b] and M_M[3]>=lyd1[b]):
                                f1a2v3+=1
                            if (M_M[4]<lzd1[c] and M_M[5]>=lzd1[c]):
                                f1a2v3+=1
                            values_1.append(f1a2v3)

                            if ref_dual:
                                if f1a2v3==0:
                                    internos1.append(e)
                                if f1a2v3==1:
                                    faces1.append(e)
                                elif f1a2v3==3:
                                    vertice=e

                        MM.mb.tag_set_data(D1_tag, elem_por_L1,values_1)

                        MM.mb.tag_set_data(fine_to_primal1_classic_tag, elem_por_L1, np.repeat(nc1,len(elem_por_L1)))

                        # Enriquece a malha dual
                        if ref_dual:
                            #viz_vert=rng.unite(rng.Range(vertice),MM.mtu.get_bridge_adjacencies(vertice, 1, 3))
                            viz_vert=MM.mtu.get_bridge_adjacencies(vertice, 1, 3)
                            cent_v=cent=MM.mtu.get_average_position([np.uint64(vertice)])
                            new_vertices=[]
                            perm1=MM.mb.tag_get_data(MM.perm_tag,viz_vert)
                            perm1_x=perm1[:,0]
                            perm1_y=perm1[:,4]
                            perm1_z=perm1[:,8]
                            r=False
                            r_p=0
                            #print(max(perm1_x)/min(perm1_x),max(perm1_y)/min(perm1_y),max(perm1_z)/min(perm1_z))
                            if max(perm1_x)>r_p*min(perm1_x) or max(perm1_y)>r_p*min(perm1_y) or max(perm1_z)>r_p*min(perm1_z):
                                r=True
                            #print(max(perm1_x)/min(perm1_x))
                            #rng.subtract(rng.Range(vertice),viz_vert)
                            for v in viz_vert:
                                cent=MM.mtu.get_average_position([np.uint64(v)])
                                if (cent[2]-cent_v[2])<0.01 and r:# and v in faces1:
                                    new_vertices.append(v)

                            adjs_new_vertices=[MM.mtu.get_bridge_adjacencies(v,2,3) for v in new_vertices]

                            new_faces=[]
                            for conj in adjs_new_vertices:
                                v=rng.intersect(rng.Range(internos1),conj)
                                if len(v)>0:
                                    new_faces.append(np.uint64(v))
                            for f in new_faces:
                                try:
                                    vfd=0
                                    #MM.mb.tag_set_data(D1_tag, f,np.repeat(1,len(f)))
                                except:
                                    import pdb; pdb.set_trace()

                        #MM.mb.tag_set_data(D1_tag, new_vertices,np.repeat(2,len(new_vertices)))
                        MM.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        MM.mb.add_child_meshset(l2_meshset,l1_meshset)
#-------------------------------------------------------------------------------

print('Criação da árvore de meshsets primais: ',time.time()-t0)

meshsets_nv1 = MM.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
meshsets_nv2 = MM.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))

n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
all_meshsets = [meshsets_nv1, meshsets_nv2]
t0 = time.time()



for i in range(n_levels):
    meshsets = all_meshsets[i]
    # names_tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
    tag_boundary = MM.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    utpy.set_faces_in_boundary_by_meshsets(MM.mb, MM.mtu, meshsets, tag_boundary)
    tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
t1 = time.time()
print('tempo faces contorno')
print(t1-t0)

for meshset in meshsets_nv2:
    nc = MM.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = MM.mb.get_entities_by_handle(meshset)
    MM.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))

internos=MM.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=MM.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=MM.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=MM.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))

MM.mb.tag_set_data(fine_to_primal1_classic_tag,vertices,np.arange(0,len(vertices)))

for meshset in meshsets_nv1:
    elems = MM.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, vertices)
    try:
        nc = MM.mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    except:
        import pdb; pdb.set_trace()
    MM.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))
    MM.mb.tag_set_data(primal_id_tag1, meshset, nc)


v=MM.mb.create_meshset()
MM.mb.add_entities(v,vertices)
ver=MM.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))
mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

for meshset in meshsets_nv2: #print(rng.intersect(M1.mb.get_entities_by_handle(meshset), ver))
    elems = MM.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, ver)
    try:
        nc = MM.mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    except:
        import pdb; pdb.set_trace()
    MM.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))
    MM.mb.tag_set_data(primal_id_tag2, meshset, nc)

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)

nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv
l_elems=[internos,faces,arestas,vertices]
l_ids=[0,nni,nnf,nne,nnv]
for i, elems in enumerate(l_elems):
    MM.mb.tag_set_data(MM.ID_reordenado_tag,elems,np.arange(l_ids[i],l_ids[i+1]))

def add_topology(conj_vols,tag_local,lista):
    all_fac=np.uint64(MM.mtu.get_bridge_adjacencies(conj_vols, 2 ,2))
    all_int_fac=np.uint64([face for face in all_fac if len(MM.mb.get_adjacencies(face, 3))==2])
    adjs=np.array([MM.mb.get_adjacencies(face, 3) for face in all_int_fac])
    adjs1=MM.mb.tag_get_data(tag_local,np.array(adjs[:,0]),flat=True)
    adjs2=MM.mb.tag_get_data(tag_local,np.array(adjs[:,1]),flat=True)
    adjsg1=MM.mb.tag_get_data(MM.ID_reordenado_tag,np.array(adjs[:,0]),flat=True)
    adjsg2=MM.mb.tag_get_data(MM.ID_reordenado_tag,np.array(adjs[:,1]),flat=True)
    Gids=MM.mb.tag_get_data(MM.ID_reordenado_tag,conj_vols,flat=True)
    lista.append(Gids)
    lista.append(all_int_fac)
    lista.append(adjs1)
    lista.append(adjs2)
    lista.append(adjsg1)
    lista.append(adjsg2)

t0=time.time()
local_id_int_tag = MM.mb.tag_get_handle("local_id_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
local_id_fac_tag = MM.mb.tag_get_handle("local_fac_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
MM.mb.tag_set_data(local_id_int_tag, MM.all_volumes,np.repeat(len(MM.all_volumes)+1,len(MM.all_volumes)))
MM.mb.tag_set_data(local_id_fac_tag, MM.all_volumes,np.repeat(len(MM.all_volumes)+1,len(MM.all_volumes)))
tags_criadas_aqui += ['local_id_internos', 'local_fac_internos']
sgids=0
li=[]
ci=[]
di=[]
cont=0

intern_adjs_by_dual=[]
faces_adjs_by_dual=[]
dual_1_meshset=MM.mb.create_meshset()
tag_dual_1_meshset = MM.mb.tag_get_handle('DUAL_1_MESHSET', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
MM.mb.tag_set_data(tag_dual_1_meshset, 0, dual_1_meshset)
tags_criadas_aqui.append('DUAL_1_MESHSET')


D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
for i in range(len(lxd1)-1):
    x0=lxd1[i]
    x1=lxd1[i+1]
    for j in range(len(lyd1)-1):
        y0=lyd1[j]
        y1=lyd1[j+1]
        for k in range(len(lzd1)-1):
            z0=lzd1[k]
            z1=lzd1[k+1]
            tb=time.time()
            box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
            vols=get_box(MM.all_volumes, all_centroids, box_dual_1, False)
            tipo=MM.mb.tag_get_data(D1_tag,vols,flat=True)
            inter=rng.Range(np.array(vols)[np.where(tipo==0)[0]])

            MM.mb.tag_set_data(local_id_int_tag,inter,range(len(inter)))
            add_topology(inter,local_id_int_tag,intern_adjs_by_dual)


            fac=rng.Range(np.array(vols)[np.where(tipo==1)[0]])
            fac_centroids=np.array([MM.mtu.get_average_position([f]) for f in fac])

            box_faces_x=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x0+lx/2,y1+ly/2,z1+lz/2]])
            box_faces_y=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y0+ly/2,z1+lz/2]])
            box_faces_z=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z0+lz/2]])

            faces_x=get_box(fac, fac_centroids, box_faces_x, False)

            faces_y=get_box(fac, fac_centroids, box_faces_y, False)
            f1=rng.unite(faces_x,faces_y)

            faces_z=get_box(fac, fac_centroids, box_faces_z, False)
            f1=rng.unite(f1,faces_z)

            if i==len(lxd1)-2:
                box_faces_x2=np.array([[x1-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_x2=get_box(fac, fac_centroids, box_faces_x2, False)
                f1=rng.unite(f1,faces_x2)

            if j==len(lyd1)-2:
                box_faces_y2=np.array([[x0-lx/2,y1-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_y2=get_box(fac, fac_centroids, box_faces_y2, False)
                f1=rng.unite(f1,faces_y2)

            if k==len(lzd1)-2:
                box_faces_z2=np.array([[x0-lx/2,y0-ly/2,z1-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_z2=get_box(fac, fac_centroids, box_faces_z2, False)
                f1=rng.unite(f1,faces_z2)

            sgids+=len(f1)
            MM.mb.tag_set_data(local_id_fac_tag,f1,range(len(f1)))
            add_topology(f1,local_id_fac_tag,faces_adjs_by_dual)

print(time.time()-t1,"criou meshset")


np.save('intern_adjs_by_dual', intern_adjs_by_dual)
np.save('faces_adjs_by_dual', faces_adjs_by_dual)



ext_h5m = input_file + '_dual_primal.h5m'
ext_vtk = input_file + '_dual_primal.vtk'
MM.mb.write_file(ext_h5m)
MM.mb.write_file(ext_vtk, [AV_meshset])

print('saiu preprocess')
