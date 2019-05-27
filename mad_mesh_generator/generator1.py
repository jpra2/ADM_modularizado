# -*- coding: utf-8 -*-


import os
from .load_mesh import Mesh
import yaml
import pdb
from definitions.dual_primal import get_box
from pymoab import types, rng, topo_util
import numpy as np
import sympy
import scipy.sparse as sp
from utils.others_utils import OtherUtils as oth


__all__ = ['mesh', 'SOL_ADM_fina', 'data_loaded']

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)
    # data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    # data_loaded = yaml.full_load(stream)

input_file = data_loaded['input_file']
ext_h5m_in = input_file + '_dual_primal.h5m'
ext_vtk_in = input_file + '_dual_primal.vtk'

# Distância, em relação ao poço, até onde se usa malha fina
r0 = data_loaded['rs']['r0']
# Distância, em relação ao poço, até onde se usa malha intermediária
r1 = data_loaded['rs']['r1']

os.chdir(flying_dir)

mesh = Mesh(ext_h5m_in)

tags2 = []
press_value_tag = mesh.mb.tag_get_handle('P', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
vazao_value_tag = mesh.mb.tag_get_handle('Q', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
mesh.tags['P'] = press_value_tag
mesh.tags['Q'] = vazao_value_tag

all_centroids = mesh.all_centroids
all_volumes = mesh.all_volumes
gravity = data_loaded['gravity']

wells_injector = mesh.mb.create_meshset()
wells_producer = mesh.mb.create_meshset()
wells_producer_tag=mesh.mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
wells_injector_tag=mesh.mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
mesh.mb.tag_set_data(wells_producer_tag, 0, wells_producer)
mesh.mb.tag_set_data(wells_injector_tag, 0, wells_injector)
mesh.tags['WELLS_PRODUCER'] = wells_producer_tag
mesh.tags['WELLS_INJECTOR'] = wells_injector_tag

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

        mesh.mb.tag_set_data(press_value_tag, volumes, pressao)

    elif  w['type_prescription'] == 'neumann':
        volumes_n += list(volumes)
        bvn.append(box_volumes)
        value = value/len(volumes)
        if w['type_well'] == 'injector':
            value = -value
        mesh.mb.tag_set_data(vazao_value_tag, volumes, np.repeat(value, len(volumes)))

    else:
        raise NameError("type_prescription == 'neumann' or 'dirichlet'")

    if w['type_well'] == 'injector':
        mesh.mb.add_entities(wells_injector, volumes)
    else:
        mesh.mb.add_entities(wells_producer, volumes)

bvfd = list()
for bvdi in bvd:
    bvfd.append(np.array([np.array([bvdi[0][0]-r0, bvdi[0][1]-r0, bvdi[0][2]-r0]), np.array([bvdi[1][0]+r0, bvdi[1][1]+r0, bvdi[1][2]+r0])]))
bvfd = np.array(bvfd)

bvfn = list()
for bvfni in bvn:
    bvfn.append(np.array([np.array([bvn[0][0]-r0, bvn[0][1]-r0, bvn[0][2]-r0]), np.array([bvn[1][0]+r0, bvn[1][1]+r0, bvn[1][2]+r0])]))
bvfn = np.array(bvfn)

bvid = list()
for bvidi in bvd:
    gg = np.array([bvidi[0]-r1, bvidi[1]+r1])
    bvid.append(gg)
bvid = np.array(bvid)

bvin = list()
for bvini in bvn:
    bvin.append(np.array([np.array([bvini[0][0]-r1, bvini[0][1]-r1, bvini[0][2]-r1]), np.array([bvini[1][0]+r1, bvini[1][1]+r1, bvini[1][2]+r1])]))
bvin = np.array(bvin)

volumes_d = rng.Range(volumes_d)
volumes_n = rng.Range(volumes_n)

# volumes finos por neumann
volumes_fn = []
for i in bvfn:
    volumes_fn += list(get_box(all_volumes, all_centroids, i, False))
volumes_fn = rng.Range(volumes_fn)

# volumes finos por Dirichlet
volumes_fd = []
for i in bvfd:
    volumes_fd += list(get_box(all_volumes, all_centroids, i, False))
volumes_fd = rng.Range(volumes_fd)

volumes_f = rng.unite(volumes_fn, volumes_fd)
D1_tag = mesh.tags['d1']
D2_tag = mesh.tags['d2']

finos = list(rng.unite(rng.unite(volumes_d, volumes_n), volumes_f))
vertices = mesh.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices = rng.unite(vertices, mesh.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids = np.array([mesh.mtu.get_average_position([v]) for v in vertices])

# volumes intermediarios por neumann
volumes_in = []
for i in bvin:
    volumes_in += list(get_box(vertices, all_vertex_centroids, i, False))
volumes_in = rng.Range(volumes_in)

# volumes intermediarios por Dirichlet
volumes_id = []
for i in bvid:
    volumes_id += list(get_box(vertices, all_vertex_centroids, i, False))
volumes_id = rng.Range(volumes_id)

intermediarios=rng.unite(volumes_id,volumes_in)
mesh.intermediarios = intermediarios
meshset_intermediarios = mesh.mb.create_meshset()
mesh.mb.add_entities(meshset_intermediarios, intermediarios)
intermediarios_tag = mesh.mb.tag_get_handle('intermediarios', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
mesh.mb.tag_set_data(intermediarios_tag, 0, meshset_intermediarios)
mesh.tags['intermediarios'] = intermediarios_tag

L1_ID_tag=mesh.mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L2_ID_tag=mesh.mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L3_ID_tag=mesh.mb.tag_get_handle("NIVEL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
mesh.tags['l1_ID'] = L1_ID_tag
mesh.tags['l2_ID'] = L2_ID_tag
mesh.tags['NIVEL_ID'] = L3_ID_tag

SOL_ADM_f = np.repeat(1,len(mesh.all_volumes))

def get_perm_lims(mesh):
    med_perm_by_primal_1=[]
    meshset_by_L2 = mesh.mb.get_child_meshsets(mesh.L2_meshset)
    for m2 in meshset_by_L2:
        meshset_by_L1=mesh.mb.get_child_meshsets(m2)
        for m1 in meshset_by_L1:
            elem_by_L1 = mesh.mb.get_entities_by_handle(m1)
            perm1=mesh.mb.tag_get_data(mesh.tags['PERM'],elem_by_L1).reshape([len(elem_by_L1),9])
            med1_x=sum(perm1[:,0])/len(perm1[:,0])
            med_perm_by_primal_1.append(med1_x)
    med_perm_by_primal_1=np.sort(med_perm_by_primal_1)
    s=1.0    #Parâmetro da secante
    lg=np.log(med_perm_by_primal_1)
    ordem=11
    print("fit")
    fit=np.polyfit(range(len(lg)),lg,ordem)
    x=sympy.Symbol('x',real=True,positive=True)
    func=0
    for i in range(ordem+1):
        func+=fit[i]*x**(ordem-i)
    print("deriv")
    derivada=sympy.diff(func,x)
    inc_secante=(lg[-1]-lg[0])/len(lg)
    print("solve")
    equa=sympy.Eq(derivada,2*inc_secante)
    #real_roots=sympy.solve(equa)
    ind_inferior=int(sympy.nsolve(equa,0.1*len(lg),verify=False))
    if ind_inferior<0:
        ind_inferior=0
    ind_superior=int(sympy.nsolve(equa,0.9*len(lg),verify=False))
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

    ind_inferior=new_ind_inferior
    ind_superior=new_ind_superior

    val_barreira=med_perm_by_primal_1[ind_inferior]
    val_canal=med_perm_by_primal_1[ind_superior]

    return val_barreira, val_canal

val_barreira, val_canal = get_perm_lims(mesh)

ares2_tag=mesh.mb.tag_get_handle("ares_2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
mesh.tags['ares_2'] = ares2_tag


def set_contraste(mesh):
    mesh.perm_tag = mesh.tags['PERM']
    meshset_by_L2 = mesh.mb.get_child_meshsets(mesh.L2_meshset)
    max1 = 0
    for m2 in meshset_by_L2:
        meshset_by_L1=mesh.mb.get_child_meshsets(m2)
        for m1 in meshset_by_L1:
            ver_1=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
            viz_vert=mesh.mtu.get_bridge_adjacencies(ver_1,1,3)
            k_vert=mesh.mb.tag_get_data(mesh.perm_tag,ver_1)[:,0]
            facs_ver1=mesh.mtu.get_bridge_adjacencies(ver_1,2,2)
            max_r=0
            vers=[]
            # somak=0
            for f in facs_ver1:
                viz_facs=mesh.mtu.get_bridge_adjacencies(f,2,3)
                ares_m=mesh.mb.create_meshset()
                mesh.mb.add_entities(ares_m,viz_facs)
                ares=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                if len(ares)>0:
                    viz_ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                    ares_m=mesh.mb.create_meshset()
                    mesh.mb.add_entities(ares_m,viz_ares)
                    ares_com_novas=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                    novas_ares=rng.subtract(ares_com_novas,ares)
                    ares=rng.unite(ares,novas_ares)
                    for i in range(20):
                        viz_ares=mesh.mtu.get_bridge_adjacencies(novas_ares,2,3)
                        ares_m=mesh.mb.create_meshset()
                        mesh.mb.add_entities(ares_m,viz_ares)
                        ares_com_novas=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                        novas_ares=rng.subtract(ares_com_novas,ares)
                        ares=rng.unite(ares,novas_ares)

                        if len(novas_ares)==0:
                            break
                    a1=ares
                    gids_ares=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],a1,flat=True)
                    facs_ares=mesh.mtu.get_bridge_adjacencies(a1,2,2)
                    # somak+= sum(mesh.mb.tag_get_data(mesh.k_eq_tag,facs_ares,flat=True))/k_vert
                    ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                    ares_m=mesh.mb.create_meshset()
                    mesh.mb.add_entities(ares_m,ares)
                    verts=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
                    ares_ares=verts=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                    v_verts=mesh.mtu.get_bridge_adjacencies(verts,2,3)
                    ares_ares=rng.unite(ares_ares,verts)
                    ares=rng.unite(ares,v_verts)
                    k_ares_max=mesh.mb.tag_get_data(mesh.perm_tag,ares)[:,0].max()
                    k_ares_min=mesh.mb.tag_get_data(mesh.perm_tag,ares)[:,0].min()
                    # k_ares_med=sum(mesh.mb.tag_get_data(mesh.perm_tag,a1)[:,0])/len(ares)
                    ver_2=np.uint64(rng.subtract(verts,ver_1))
                    k_ver2=mesh.mb.tag_get_data(mesh.perm_tag,ver_2)[0][0]
                    vers.append(ver_2)
                    perm_ares=mesh.mb.tag_get_data(mesh.perm_tag,ares)[:,0]
                    try:
                        r_ver=mesh.mb.tag_get_data(ares2_tag,ver_1)
                        r_k_are_ver=float(max((k_ares_max-k_ares_min)/min(k_vert,k_ver2),r_ver))
                    except:
                        r_k_are_ver=float((k_ares_max-k_ares_min)/min(k_vert,k_ver2))
                    r_k_are_ver=max(r_k_are_ver,max(k_vert,k_ver2)/perm_ares.min())
                    r_k_are_ver=max(r_k_are_ver,perm_ares.max()/min(k_vert,k_ver2))
                    if r_k_are_ver>max_r:
                        max_r=r_k_are_ver

                    #mesh.mb.tag_set_data(ares2_tag, ares, np.repeat(float((k_ares_max-k_ares_min)/k_vert),len(ares)))
                    mesh.mb.tag_set_data(ares2_tag, ver_1,r_k_are_ver)
            perm_viz=mesh.mb.tag_get_data(mesh.tags['PERM'], viz_vert)[:,0]
            # raz=float(k_vert/perm_viz.min())
            # mesh.mb.tag_set_data(raz_tag, ver_1,raz)
            # mesh.mb.tag_set_data(var_tag, ver_1,var)
            # mesh.mb.tag_set_data(var2_tag, ver_1,somak)
            if max_r>200:
                for v in vers:
                    try:
                        r_ver=mesh.mb.tag_get_data(ares2_tag,v)
                    except:
                        r_ver=0
                    #mesh.mb.tag_set_data(ares2_tag, v,float(max(max_r/4,r_ver)))
                    #k_ver2=mesh.mb.tag_get_data(mesh.perm_tag,ver_2)[:,0]
                    #try:
                    #    r_ver=mesh.mb.tag_get_data(ares2_tag,ver_2)
                    #    r_k_are_ver=float(max((k_ares_max-k_ares_min)/k_ver2,r_ver))
                    #except:
                    #    r_k_are_ver=float((k_ares_max-k_ares_min)/k_ver2)
                    #mesh.mb.tag_set_data(ares2_tag, ver_2,r_k_are_ver)
    del mesh.perm_tag

set_contraste(mesh)

def geracao_adm1_mesh(mesh):
    n1 = 0
    n2 = 0
    aux = 0
    var_tag=mesh.mb.tag_get_handle("var", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mesh.tags['var'] = var_tag
    vertices = mesh.wirebasket_elems[0][3]
    gids=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], vertices, flat=True)
    diagon=np.array(abs(mesh.matrices['Tf'][gids].min(axis=1)).toarray().transpose()[0])
    mesh.mb.tag_set_data(var_tag,vertices,diagon)
    meshset_by_L2 = mesh.mb.get_child_meshsets(mesh.L2_meshset)
    for m2 in meshset_by_L2:
        tem_poço_no_vizinho=False
        meshset_by_L1=mesh.mb.get_child_meshsets(m2)
        for m1 in meshset_by_L1:
            elem_by_L1 = mesh.mb.get_entities_by_handle(m1)
            ver_1=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
            ver_1=rng.unite(ver_1,mesh.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))

            if ver_1[0] in finos:
                aux=1
                tem_poço_no_vizinho=True
            else:
                # viz_vertice=mesh.mtu.get_bridge_adjacencies(ver_1,2,3)
                # k_vert=float(mesh.mb.tag_get_data(mesh.perm_tag,ver_1)[:,0])
                # k_viz=mesh.mb.tag_get_data(mesh.perm_tag,viz_vertice)[:,0]
                # raz=float(mesh.mb.tag_get_data(raz_tag,ver_1))
                perm1=mesh.mb.tag_get_data(mesh.tags['PERM'],elem_by_L1)
                med1_x=sum(perm1[:,0])/len(perm1[:,0])
                # med1_y=sum(perm1[:,4])/len(perm1[:,4])
                # med1_z=sum(perm1[:,8])/len(perm1[:,8])
                # gids_primal=mesh.mb.tag_get_data(mesh.ID_reordenado_tag,elem_by_L1)
                # press_primal=SOL_ADM_f[gids_primal]
                # ares=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([2]))
                #
                # viz_ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                # ares_m=mesh.mb.create_meshset()
                # mesh.mb.add_entities(ares_m,viz_ares)
                # viz_ares_ares=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                # ares=viz_ares_ares
                #
                # viz_ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                # ares_m=mesh.mb.create_meshset()
                # mesh.mb.add_entities(ares_m,viz_ares)
                # viz_ares_ares=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                # viz_ares_ver=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
                # viz_ares_fac=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([1]))
                # viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_ver)
                # viz_ares_ares=rng.unite(viz_ares_ares,ver_1)
                # viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_fac)
                # ares=viz_ares_ares

                # k_ares_max=mesh.mb.tag_get_data(mesh.perm_tag,ares)[:,0].max()
                # k_ares_min=mesh.mb.tag_get_data(mesh.perm_tag,ares)[:,0].min()
                # r_k_are_ver=float((k_ares_max-k_ares_min)/k_vert)
                #mesh.mb.tag_set_data(ares_tag, ares, np.repeat(r_k_are_ver,len(ares)))
                r_k_are_ver=float(mesh.mb.tag_get_data(ares2_tag,ver_1))
                #
                # #
                var=float(mesh.mb.tag_get_data(var_tag,ver_1))
                # var2=float(mesh.mb.tag_get_data(var2_tag,ver_1))
                # ar=float(mesh.mb.tag_get_data(ares_tag,ver_1))
                # ar3=float(mesh.mb.tag_get_data(ares3_tag,ver_1))
                # ar4=mesh.mb.tag_get_data(ares4_tag,mesh.mtu.get_bridge_adjacencies(elem_by_L1,2,3),flat=True).max()
                # #ar5=float(mesh.mb.tag_get_data(ares5_tag,ver_1))
                # ar6=float(mesh.mb.tag_get_data(ares6_tag,ver_1))
                #
                # viz_meshset=mesh.mtu.get_bridge_adjacencies(elem_by_L1,2,3)
                # so_viz=rng.subtract(viz_meshset,elem_by_L1)
                # m7=max(mesh.mb.tag_get_data(ares6_tag,so_viz,flat=True))
                # m8=max(mesh.mb.tag_get_data(var_tag,so_viz,flat=True))
                # a7=ar6/var
                # a8=ar4*(1+a7)**3
                # if first:
                #     a9=1.0
                # else:
                #     a9=float(mesh.mb.tag_get_data(ares9_tag,ver_1))
                # mesh.mb.tag_set_data(ares7_tag,ver_1,a7)
                # mesh.mb.tag_set_data(ares8_tag,elem_by_L1,np.repeat(ar4/a9,len(elem_by_L1)))
                # mesh.mb.tag_set_data(ares9_tag,elem_by_L1,np.repeat(a9,len(elem_by_L1)))
                #
                # if first:
                #     max_grad=0
                # else:
                #     max_grad=get_max_grad(m1)
                # mesh.mb.tag_set_data(grad_tag,elem_by_L1,np.repeat(max_grad/grad_p_res,len(elem_by_L1)))
                # #
                #
                if med1_x<val_barreira or med1_x>val_canal or var<1 or r_k_are_ver>100:
                    aux=1
                    tem_poço_no_vizinho=True
            if ver_1[0] in intermediarios:
                tem_poço_no_vizinho=True
            if aux==1:
                aux=0
                for elem in elem_by_L1:
                    n1+=1
                    n2+=1
                    mesh.mb.tag_set_data(L1_ID_tag, elem, n1)
                    mesh.mb.tag_set_data(L2_ID_tag, elem, n2)
                    mesh.mb.tag_set_data(L3_ID_tag, elem, 1)
                    finos.append(elem)
        if tem_poço_no_vizinho==False:
            elem_by_L2 = mesh.mb.get_entities_by_handle(m2)
            if first:
                max_grad=0
            else:
                max_grad=get_max_grad(m2)

            vers=mesh.mb.get_entities_by_type_and_tag(m2, types.MBHEX, np.array([D1_tag]), np.array([3]))
            r_k_are_ver=mesh.mb.tag_get_data(ares2_tag,vers,flat=True)

            if max_grad>3*grad_p_res or max(r_k_are_ver)>1000:
                tem_poço_no_vizinho=True
        if tem_poço_no_vizinho:
            for m1 in meshset_by_L1:
                elem_by_L1 = mesh.mb.get_entities_by_handle(m1)
                n1+=1
                n2+=1
                t=1
                ver_1=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
                ver_1=rng.unite(ver_1,mesh.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
                if ver_1[0] not in finos:
                    mesh.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                    mesh.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                    mesh.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(2,len(elem_by_L1)))
                    t=0
                n1-=t
                n2-=t
        else:
            n2+=1
            for m1 in meshset_by_L1:
                elem_by_L1 = mesh.mb.get_entities_by_handle(m1)
                n1+=1
                mesh.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                mesh.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                mesh.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(3,len(elem_by_L1)))

    tags = [L1_ID_tag, L2_ID_tag]
    for tag in tags:
        try:
            all_gids = mesh.mb.tag_get_data(tag, mesh.all_volumes, flat=True)
        except:
            pdb.set_trace()
        minim = min(all_gids)
        all_gids -= minim
        mesh.mb.tag_set_data(tag, mesh.all_volumes, all_gids)

    return n1, n2

n1, n2 = geracao_adm1_mesh(mesh)

vv = mesh.mb.create_meshset()
mesh.mb.add_entities(vv, mesh.all_volumes)

def get_OP1_ADM(mesh, n1, n2):
    vertices = mesh.wirebasket_elems[0][3]
    ids_1=mesh.mb.tag_get_data(L1_ID_tag,vertices,flat=True)
    ids_class=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL1_CLASSIC'],vertices,flat=True)
    vm=mesh.mv
    PAD=mesh.matrices['OP1_AMS'].copy()
    nivel_0=mesh.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))

    ID_global1=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],nivel_0, flat=True)
    # elems_comp = rng.subtrac(mesh.all_volumes, nivel_0)
    # ids_reord_elems_comp = mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], elems_comp, flat=True)

    IDs_ADM1=mesh.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
    IDs_AMS1=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL1_CLASSIC'],nivel_0, flat=True)
    # PAD[ID_global1] = sp.csc_matrix((1,PAD.shape[1]))
    complementar = np.setdiff1d(np.arange(len(mesh.all_volumes)), ID_global1)
    data = np.ones(len(complementar))
    lines = complementar
    cols = range(len(complementar))
    permut = sp.csc_matrix((data ,(lines , cols)), shape=(len(mesh.all_volumes),len(complementar)))
    OP3 = PAD[complementar]
    OP3 = permut*OP3
    IDs_ADM1=mesh.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
    IDs_ADM_1=mesh.mb.tag_get_data(L1_ID_tag,vertices, flat=True)
    IDs_AMS_1=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL1_CLASSIC'],vertices, flat=True)
    lp=IDs_AMS_1
    cp=IDs_ADM_1
    dp=np.repeat(1,len(lp))
    permut=sp.csc_matrix((dp,(lp,cp)),shape=(len(vertices),n1))
    opad3=OP3*permut
    m=sp.find(opad3)
    l1=m[0]
    c1=m[1]
    d1=m[2]
    l1=np.concatenate([l1,ID_global1])
    c1=np.concatenate([c1,IDs_ADM1])
    d1=np.concatenate([d1,np.ones(len(nivel_0))])
    opad3=sp.csc_matrix((d1,(l1,c1)),shape=(len(mesh.all_volumes),n1))
    # print("opad1",tor-time.time(),time.time()-ta1, time.time()-tempo0_ADM)
    OP_ADM=sp.csc_matrix(opad3)
    return OP_ADM

mesh.matrices['OP1_ADM'] = get_OP1_ADM(mesh, n1, n2)

def get_OR1_ADM(mesh, n1):
    l1 = mesh.mb.tag_get_data(L1_ID_tag, mesh.all_volumes, flat=True)
    c1 = mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], mesh.all_volumes, flat=True)
    d1 = np.ones(len(l1))
    OR1_ADM = sp.csc_matrix((d1, (l1, c1)), shape=(n1, len(mesh.all_volumes)))
    return OR1_ADM

mesh.matrices['OR1_ADM'] = get_OR1_ADM(mesh, n1)

def get_OR2_ADM(mesh, n1, n2):
    l2 = mesh.mb.tag_get_data(L2_ID_tag, mesh.all_volumes, flat=True)
    c2 = mesh.mb.tag_get_data(L1_ID_tag, mesh.all_volumes, flat=True)
    d2 = np.ones(len(l2))
    OR_ADM_2 = sp.csc_matrix((d2,(l2,c2)),shape=(n2,n1))
    return OR_ADM_2

mesh.matrices['OR2_ADM'] = get_OR2_ADM(mesh, n1, n2)

def get_OP2_ADM(mesh, n1, n2):

    OP3 = mesh.matrices['OP2_AMS'].copy()

    nivel_0=mesh.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    nivel_1=mesh.mb.get_entities_by_type_and_tag(mesh.mv, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
    nivel_0_e_1=rng.unite(nivel_0,nivel_1)

    nivel_0v=mesh.mb.get_entities_by_type_and_tag(mesh.mv, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    nivel_0_e_1_v=rng.unite(nivel_0v,nivel_1)
    IDs_AMS1=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL1_CLASSIC'], nivel_0_e_1_v, flat=True)
    OP3[IDs_AMS1]=sp.csc_matrix((1,OP3.shape[1]))

    IDs_ADM_2=mesh.mb.tag_get_data(L2_ID_tag, mesh.wirebasket_elems[1][3], flat=True)
    IDs_AMS_2=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL2_CLASSIC'],mesh.wirebasket_elems[1][3], flat=True)
    lp=IDs_AMS_2
    cp=IDs_ADM_2
    dp=np.repeat(1,len(lp))
    permutc=sp.csc_matrix((dp,(lp,cp)),shape=(len(mesh.wirebasket_elems[1][3]),n2))
    opad3=OP3*permutc

    IDs_ADM_1=mesh.mb.tag_get_data(L1_ID_tag,vertices, flat=True)
    IDs_AMS_1=mesh.mb.tag_get_data(mesh.tags['FINE_TO_PRIMAL1_CLASSIC'], mesh.wirebasket_elems[0][3], flat=True)

    lp=IDs_ADM_1
    cp=IDs_AMS_1
    dp=np.repeat(1,len(lp))
    permutl=sp.csc_matrix((dp,(lp,cp)),shape=(n1,len(vertices)))
    opad3=permutl*opad3

    m=sp.find(opad3)
    l1=m[0]
    c1=m[1]
    d1=m[2]

    ID_global1=mesh.mb.tag_get_data(L1_ID_tag,nivel_0_e_1, flat=True)
    IDs_ADM1=mesh.mb.tag_get_data(L2_ID_tag,nivel_0_e_1, flat=True)

    l1=np.concatenate([l1,ID_global1])
    c1=np.concatenate([c1,IDs_ADM1])
    d1=np.concatenate([d1,np.ones(len(nivel_0_e_1))])
    opad3=sp.csc_matrix((d1,(l1,c1)),shape=(n1,n2))

    return opad3

mesh.matrices['OP2_ADM'] = get_OP2_ADM(mesh, n1, n2)

values_d = mesh.mb.tag_get_data(mesh.tags['P'], volumes_d, flat=True)
values_n = mesh.mb.tag_get_data(mesh.tags['Q'], volumes_n, flat=True)

Tf2 = mesh.matrices['Tf'].copy()
b2 = mesh.matrices['b'].copy()

ids_d = mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], volumes_d, flat=True)
Tf2, b = oth.set_boundary_dirichlet_matrix_v02(ids_d, values_d, b2, Tf2)

if len(volumes_n) > 0:
    ids_n = mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], volumes_n, flat=True)
    b2 = oth.set_boundary_neumann_v02(ids_n, values_n, b2)


def get_solution_adm(mesh, Tf2, b2):

    TADM1 = mesh.matrices['OR1_ADM']*Tf2*mesh.matrices['OP1_ADM']
    TADM2 = mesh.matrices['OR2_ADM']*TADM1*mesh.matrices['OP2_ADM']
    bADM1 = mesh.matrices['OR1_ADM']*b2
    bADM2 = mesh.matrices['OR2_ADM']*bADM1
    PMS = sp.linalg.spsolve(TADM2, bADM2)
    Pms = mesh.matrices['OP2_ADM']*PMS
    Pms = mesh.matrices['OP1_ADM']*PMS

    return Pms

pms = get_solution_adm(mesh, Tf2, b2)
SOL_ADM_fina = pms
pms_tag = mesh.mb.tag_get_handle('PMS2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
mesh.tags['PMS2'] = pms_tag
wire_elems = []
for elems in mesh.wirebasket_elems[0]:
    wire_elems += list(elems)

mesh.mb.tag_set_data(pms_tag, wire_elems, pms)
os.chdir(flying_dir)
np.save('SOL_ADM_fina', pms)
mesh.mb.write_file('malha_adm_1.vtk', [vv])
mesh.vv = vv
mesh.volumes_f = volumes_f
