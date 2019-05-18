import time
import numpy as np
from utils import pymoab_utils as utpy
from pymoab import types, rng
import scipy.sparse as sp
from utils.others_utils import OtherUtils as oth
import pdb

__all__ = ['DualPrimal']

class DualPrimal:

    def __init__(self, MM, Lx, Ly, Lz, mins, l2, l1, dx0, dy0, dz0, lx, ly, lz, data_loaded):
        gdp = GenerateDualPrimal()
        gdp.DualPrimal2(MM, Lx, Ly, Lz, mins, l2, l1, dx0, dy0, dz0)
        gdp.topology(MM, lx, ly, lz, Lx, Ly, Lz)
        gdp.get_adjs_volumes(MM)
        gdp.get_Tf(MM, data_loaded)

        self.tags = gdp.tags
        self.intern_adjs_by_dual = gdp.intern_adjs_by_dual
        self.faces_adjs_by_dual = gdp.faces_adjs_by_dual
        self.wirebasket_elems = gdp.wirebasket_elems
        self.wirebasket_numbers = gdp.wirebasket_numbers
        self.As = gdp.As


def get_box(conjunto, all_centroids, limites, return_inds):
    # conjunto-> lista
    # all_centroids->coordenadas dos centroides do conjunto
    # limites-> diagonal que define os volumes objetivo (numpy array com duas coordenadas)
    # Retorna os volumes pertencentes a conjunto cujo centroide está dentro de limites
    inds0 = np.where(all_centroids[:,0] > limites[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > limites[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > limites[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < limites[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < limites[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < limites[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols = list(c1 & c2)
    if return_inds:
        return (rng.Range(np.array(conjunto)[inds_vols]),inds_vols)
    else:
        return rng.Range(np.array(conjunto)[inds_vols])

def add_topology(MM, conj_vols,tag_local,lista):
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

def Min_Max(e, MM):
    verts = MM.mb.get_connectivity(e)
    coords = MM.mb.get_coords(verts).reshape([len(verts),3])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return([mins[0],maxs[0],mins[1],maxs[1],mins[2],maxs[2]])

class GenerateDualPrimal:

    def __init__(self):
        self.tags = dict()

    def DualPrimal2(self, MM, Lx, Ly, Lz, mins, l2, l1, dx0, dy0, dz0):
        t0 = time.time()

        tags1 = []

        D1_tag=MM.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        D2_tag=MM.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.tags['d1'] = D1_tag
        self.tags['d2'] = D2_tag
        ##########################################################################################
        tags1 += ['d1', 'd2']

        fine_to_primal1_classic_tag = MM.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        fine_to_primal2_classic_tag = MM.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        AV_meshset=MM.mb.create_meshset()
        tags1 += ['FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC']
        self.tags['FINE_TO_PRIMAL1_CLASSIC'] = fine_to_primal1_classic_tag
        self.tags['FINE_TO_PRIMAL2_CLASSIC'] = fine_to_primal2_classic_tag

        primal_id_tag1 = MM.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        primal_id_tag2 = MM.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        tags1 += ['PRIMAL_ID_1', 'PRIMAL_ID_2']
        self.tags['PRIMAL_ID_1'] = primal_id_tag1
        self.tags['PRIMAL_ID_2'] = primal_id_tag2

        L2_meshset=MM.mb.create_meshset()       # root Meshset
        l2_meshset_tag = MM.mb.tag_get_handle('L2_MESHSET', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        MM.mb.tag_set_data(l2_meshset_tag, 0, L2_meshset)
        tags1.append('L2_MESHSET')
        self.tags['L2_MESHSET'] = l2_meshset_tag

        lx2, ly2, lz2 = [], [], []
        # O valor 0.01 é adicionado para corrigir erros de ponto flutuante
        for i in range(int(Lx/l2[0])):    lx2.append(mins[0]+i*l2[0])
        for i in range(int(Ly/l2[1])):    ly2.append(mins[1]+i*l2[1])
        for i in range(int(Lz/l2[2])):    lz2.append(mins[2]+i*l2[2])
        lx2.append(Lx)
        ly2.append(Ly)
        lz2.append(Lz)
        self.lx2 = lx2
        self.ly2 = ly2
        self.lz2 = lz2

        lx1, ly1, lz1 = [], [], []
        for i in range(int(l2[0]/l1[0])):   lx1.append(i*l1[0])
        for i in range(int(l2[1]/l1[1])):   ly1.append(i*l1[1])
        for i in range(int(l2[2]/l1[2])):   lz1.append(i*l1[2])

        D_x = max(Lx-int(Lx/l1[0])*l1[0], Lx-int(Lx/l2[0])*l2[0])
        D_y = max(Ly-int(Ly/l1[1])*l1[1], Ly-int(Ly/l2[1])*l2[1])
        D_z = max(Lz-int(Lz/l1[2])*l1[2], Lz-int(Lz/l2[2])*l2[2])
        self.D_x = D_x
        self.D_y = D_y
        self.D_z = D_z
        nD_x = int((D_x+0.001)/l1[0])
        nD_y = int((D_y+0.001)/l1[1])
        nD_z = int((D_z+0.001)/l1[2])

        lxd1 = [mins[0]+dx0/100]
        for i in range(int(Lx/l1[0])-2-nD_x):
            lxd1.append(l1[0]/2+(i+1)*l1[0])
        lxd1.append(mins[0]+Lx-dx0/100)
        self.lxd1 = lxd1

        lyd1 = [mins[1]+dy0/100]
        for i in range(int(Ly/l1[1])-2-nD_y):
            lyd1.append(l1[1]/2+(i+1)*l1[1])
        lyd1.append(mins[1]+Ly-dy0/100)
        self.lyd1 = lyd1

        lzd1 = [mins[2]+dz0/100]
        for i in range(int(Lz/l1[2])-2-nD_z):
            lzd1.append(l1[2]/2+(i+1)*l1[2])
        lzd1.append(mins[2]+Lz-dz0/100)
        self.lzd1 = lzd1

        print("definiu planos do nível 1")
        lxd2 = [lxd1[0]]
        for i in range(1,int(len(lxd1)*l1[0]/l2[0])-1):
            lxd2.append(lxd1[int(i*l2[0]/l1[0] + 1e-9)+1])
        lxd2.append(lxd1[-1])
        self.lxd2 = lxd2

        lyd2 = [lyd1[0]]
        for i in range(1,int(len(lyd1)*l1[1]/l2[1])-1):
            lyd2.append(lyd1[int(i*l2[1]/l1[1]+1e-9)+1])
        lyd2.append(lyd1[-1])
        self.lyd2 = lyd2

        lzd2 = [lzd1[0]]
        for i in range(1,int(len(lzd1)*l1[2]/l2[2])-1):
            lzd2.append(lzd1[int(i*l2[2]/l1[2] + 1e-9)+1])
        lzd2.append(lzd1[-1])
        self.lzd2 = lzd2

        print("definiu planos do nível 2")

        centroids = MM.all_centroids

        nc1=0
        nc2=0
        sx=0
        ref_dual=False
        MM.mb.add_entities(AV_meshset,MM.all_volumes)
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
                                    M_M=Min_Max(e, MM)
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
                                    perMM=MM.mb.tag_get_data(MM.perm_tag,viz_vert)
                                    perMM_x=perMM[:,0]
                                    perMM_y=perMM[:,4]
                                    perMM_z=perMM[:,8]
                                    r=False
                                    r_p=0
                                    #print(max(perMM_x)/min(perMM_x),max(perMM_y)/min(perMM_y),max(perMM_z)/min(perMM_z))
                                    if max(perMM_x)>r_p*min(perMM_x) or max(perMM_y)>r_p*min(perMM_y) or max(perMM_z)>r_p*min(perMM_z):
                                        r=True
                                    #print(max(perMM_x)/min(perMM_x))
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

        t1 = time.time()
        dt = t1-t0
        print('Criação da árvore de meshsets primais: ',dt)

        meshsets_nv1, meshsets_nv2 = self.set_bound_faces_primals(MM)
        self.meshsets_nv1 = meshsets_nv1
        self.meshsets_nv2 = meshsets_nv2

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
        MM.mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

        for meshset in meshsets_nv2: #print(rng.intersect(MM.mb.get_entities_by_handle(meshset), ver))
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
        l_elems=[internos, faces , arestas, vertices]
        l_ids=[0,nni,nnf,nne,nnv]
        wire_numbers1 = [ni, nf, na, nv]
        for i, elems in enumerate(l_elems):
            MM.mb.tag_set_data(MM.ID_reordenado_tag,elems,np.arange(l_ids[i],l_ids[i+1]))

        self.wirebasket_elems = []
        self.wirebasket_numbers = []
        self.wirebasket_elems.append(l_elems)
        self.wirebasket_numbers.append(wire_numbers1)

    def set_bound_faces_primals(self, MM):
        t0 = time.time()
        meshsets_nv1 = MM.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([self.tags['PRIMAL_ID_1']]), np.array([None]))
        meshsets_nv2 = MM.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([self.tags['PRIMAL_ID_2']]), np.array([None]))

        n_levels = 2
        name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
        all_meshsets = [meshsets_nv1, meshsets_nv2]
        t2 = time.time()

        for i in range(n_levels):
            meshsets = all_meshsets[i]
            # names_tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
            tag_boundary = MM.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
            self.tags[name_tag_faces_boundary_meshsets + str(i+2)] = tag_boundary
            utpy.set_faces_in_boundary_by_meshsets(MM.mb, MM.mtu, meshsets, tag_boundary)

        t3 = time.time()
        dt2 = t3 - t0

        return meshsets_nv1, meshsets_nv2

    def topology(self, MM, lx, ly, lz, Lx, Ly, Lz):
        t0 = time.time()
        local_id_int_tag = MM.mb.tag_get_handle("local_id_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        local_id_fac_tag = MM.mb.tag_get_handle("local_fac_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        MM.mb.tag_set_data(local_id_int_tag, MM.all_volumes,np.repeat(len(MM.all_volumes)+1,len(MM.all_volumes)))
        MM.mb.tag_set_data(local_id_fac_tag, MM.all_volumes,np.repeat(len(MM.all_volumes)+1,len(MM.all_volumes)))
        self.tags['local_id_internos'] = local_id_int_tag
        self.tags['local_fac_internos'] = local_id_fac_tag
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
        self.tags['DUAL_1_MESHSET'] = tag_dual_1_meshset

        D_x = self.D_x
        D_y = self.D_y
        D_z = self.D_z
        lxd1 = self.lxd1
        lyd1 = self.lyd1
        lzd1 = self.lzd1
        D1_tag = self.tags['d1']
        all_centroids = MM.all_centroids

        # for i in range(len(lxd1)-1):
        #     x0=lxd1[i]
        #     x1=lxd1[i+1]
        #     for j in range(len(lyd1)-1):
        #         y0=lyd1[j]
        #         y1=lyd1[j+1]
        #         for k in range(len(lzd1)-1):
        #             z0=lzd1[k]
        #             z1=lzd1[k+1]
        #             tb=time.time()
        #             box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
        #             vols=get_box(MM.all_volumes, all_centroids, box_dual_1, False)
        #             tipo=MM.mb.tag_get_data(D1_tag,vols,flat=True)
        #             inter=rng.Range(np.array(vols)[np.where(tipo==0)[0]])
        #
        #             MM.mb.tag_set_data(local_id_int_tag,inter,range(len(inter)))
        #             add_topology(MM, inter,local_id_int_tag,intern_adjs_by_dual)
        #
        #             fac=rng.Range(np.array(vols)[np.where(tipo==1)[0]])
        #             fac_centroids=np.array([MM.mtu.get_average_position([f]) for f in fac])
        #
        #             box_faces_x=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x0+lx/2,y1+ly/2,z1+lz/2]])
        #             box_faces_y=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y0+ly/2,z1+lz/2]])
        #             box_faces_z=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z0+lz/2]])
        #
        #             faces_x=get_box(fac, fac_centroids, box_faces_x, False)
        #
        #             faces_y=get_box(fac, fac_centroids, box_faces_y, False)
        #             f1=rng.unite(faces_x,faces_y)
        #
        #             faces_z=get_box(fac, fac_centroids, box_faces_z, False)
        #             f1=rng.unite(f1,faces_z)
        #
        #             if i==len(lxd1)-2:
        #                 box_faces_x2=np.array([[x1-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
        #                 faces_x2=get_box(fac, fac_centroids, box_faces_x2, False)
        #                 f1=rng.unite(f1,faces_x2)
        #
        #             if j==len(lyd1)-2:
        #                 box_faces_y2=np.array([[x0-lx/2,y1-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
        #                 faces_y2=get_box(fac, fac_centroids, box_faces_y2, False)
        #                 f1=rng.unite(f1,faces_y2)
        #
        #             if k==len(lzd1)-2:
        #                 box_faces_z2=np.array([[x0-lx/2,y0-ly/2,z1-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
        #                 faces_z2=get_box(fac, fac_centroids, box_faces_z2, False)
        #                 f1=rng.unite(f1,faces_z2)
        #
        #             sgids+=len(f1)
        #             MM.mb.tag_set_data(local_id_fac_tag,f1,range(len(f1)))
        #             add_topology(MM, f1,local_id_fac_tag,faces_adjs_by_dual)
        # t1 = time.time()
        # dt = t1-t0
        # print(time.time()-t0,"criou meshset")
        xmin = 0.0
        ymin = 0.0
        zmin = 0.0
        xmax = Lx
        ymax = Ly
        zmax = Lz

        for i in range(len(lxd1)-1):
            x0=lxd1[i]
            x1=lxd1[i+1]
            box_x=np.array([[x0-0.01,ymin,zmin],[x1+0.01,ymax,zmax]])
            vols_x=get_box(MM.all_volumes, all_centroids, box_x, False)
            x_centroids=np.array([MM.mtu.get_average_position([v]) for v in vols_x])
            for j in range(len(lyd1)-1):
                y0=lyd1[j]
                y1=lyd1[j+1]
                box_y=np.array([[x0-0.01,y0-0.01,zmin],[x1+0.01,y1+0.01,zmax]])
                vols_y=get_box(vols_x, x_centroids, box_y, False)
                y_centroids=np.array([MM.mtu.get_average_position([v]) for v in vols_y])
                for k in range(len(lzd1)-1):
                    z0=lzd1[k]
                    z1=lzd1[k+1]
                    tb=time.time()
                    box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
                    vols=get_box(vols_y, y_centroids, box_dual_1, False)
                    tipo=MM.mb.tag_get_data(D1_tag,vols,flat=True)
                    inter=rng.Range(np.array(vols)[np.where(tipo==0)[0]])

                    MM.mb.tag_set_data(local_id_int_tag,inter,range(len(inter)))
                    add_topology(MM, inter,local_id_int_tag,intern_adjs_by_dual)


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
                    add_topology(MM, f1,local_id_fac_tag,faces_adjs_by_dual)

        print(time.time()-t0,"criou meshset")

        self.intern_adjs_by_dual = intern_adjs_by_dual
        self.faces_adjs_by_dual = faces_adjs_by_dual

    def get_adjs_volumes(self, MM):
        MM.all_intern_faces = rng.subtract(MM.all_faces, MM.all_faces_boundary)
        MM.all_intern_adjacencies = np.array([np.array(MM.mb.get_adjacencies(face, 3)) for face in MM.all_intern_faces])
        MM.all_adjacent_volumes=[]
        MM.all_adjacent_volumes.append(MM.mb.tag_get_data(MM.ID_reordenado_tag, np.array(MM.all_intern_adjacencies[:,0]), flat=True))
        MM.all_adjacent_volumes.append(MM.mb.tag_get_data(MM.ID_reordenado_tag, np.array(MM.all_intern_adjacencies[:,1]), flat=True))

    def get_Tf_dep0(self, MM, *args, **kwargs):

        ni = self.wirebasket_numbers[0][0]
        nf = self.wirebasket_numbers[0][1]
        ne = self.wirebasket_numbers[0][2]
        nv = self.wirebasket_numbers[0][3]

        nni = self.wirebasket_numbers[0][0]
        nnf = self.wirebasket_numbers[0][1] + nni
        nne = self.wirebasket_numbers[0][2] + nnf
        nnv = self.wirebasket_numbers[0][3] + nne

        ADJs1 = MM.all_adjacent_volumes[0]
        ADJs2 = MM.all_adjacent_volumes[1]
        ks = MM.mb.tag_get_data(MM.k_eq_tag, MM.all_intern_faces,flat=True)
        lines = ADJs1
        cols = ADJs2
        data = ks

        Ttpfa = sp.csc_matrix((data,(lines,cols)),shape=(len(MM.all_volumes),len(MM.all_volumes)))
        rr = -np.array(Ttpfa.sum(axis=1))[:,0]
        soma = sp.csc_matrix((rr,(range(Ttpfa.shape[0]),range(Ttpfa.shape[0]))),shape=(len(MM.all_volumes),len(MM.all_volumes)))
        Ttpfa += soma

        #internos
        Aii = Ttpfa[0:nni, 0:nni]
        Aif = Ttpfa[0:nni, nni:nnf]

        #faces
        Aff = Ttpfa[nni:nnf, nni:nnf]
        Afe = Ttpfa[nni:nnf, nnf:nne]
        rr = -np.array(Aif.transpose().sum(axis=1))[:,0]
        soma = sp.csc_matrix((rr,(range(Aff.shape[0]),range(Aff.shape[0]))),shape=(Aff.shape))
        Aff += soma

        # d1 = np.matrix(Aff.diagonal()).reshape([nf, 1])
        # d1 += soma

        #arestas
        Aee = Ttpfa[nnf:nne, nnf:nne]
        Aev = Ttpfa[nnf:nne, nne:nnv]
        rr = -np.array(Afe.transpose().sum(axis=1))[:,0]
        soma = sp.csc_matrix((rr,(range(Aee.shape[0]),range(Aee.shape[0]))),shape=(Aee.shape))
        Aee += soma
        # d1 += soma
        # Aee.setdiag(d1)
        Ivv = sp.identity(nv)

        As = {}
        As['Aii'] = Aii
        As['Aif'] = Aif
        As['Aff'] = Aff
        As['Afe'] = Afe
        As['Aee'] = Aee
        As['Aev'] = Aev
        As['Ivv'] = Ivv
        As['T'] = Ttpfa
        self.As = As

    def get_Tf(self, MM, data_loaded):

        b = np.zeros(len(MM.all_volumes))

        gamaf_tag = MM.mb.tag_get_handle('GAMAF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.tags['GAMAF'] = gamaf_tag
        sgravf_tag = MM.mb.tag_get_handle('SGRAVF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.tags['SGRAVF'] = sgravf_tag
        sgravv_tag = MM.mb.tag_get_handle('SGRAVV', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.tags['SGRAVV'] = sgravv_tag
        gama = float(data_loaded['dados_monofasico']['gama'])
        MM.mb.tag_set_data(gamaf_tag, MM.all_faces, np.repeat(gama, len(MM.all_faces)))

        # ni = self.wirebasket_numbers[0][0]
        # nf = self.wirebasket_numbers[0][1]
        # ne = self.wirebasket_numbers[0][2]
        # nv = self.wirebasket_numbers[0][3]
        #
        # nni = self.wirebasket_numbers[0][0]
        # nnf = self.wirebasket_numbers[0][1] + nni
        # nne = self.wirebasket_numbers[0][2] + nnf
        # nnv = self.wirebasket_numbers[0][3] + nne

        ADJs1 = MM.all_adjacent_volumes[0]
        ADJs2 = MM.all_adjacent_volumes[1]
        ids_volumes0 = MM.mb.tag_get_data(MM.tags['IDS_VOLUMES'], np.array(MM.all_intern_adjacencies[:,0]), flat=True)
        ids_volumes1 = MM.mb.tag_get_data(MM.tags['IDS_VOLUMES'], np.array(MM.all_intern_adjacencies[:,1]), flat=True)
        ks = MM.mb.tag_get_data(MM.k_eq_tag, MM.all_intern_faces, flat=True)
        lines = []
        cols = []
        data = []

        s_gravf = np.zeros(len(MM.all_intern_faces))

        for i, face in enumerate(MM.all_intern_faces):
            gid0 = ADJs1[i]
            gid1 = ADJs2[i]
            keq = ks[i]
            id0 = ids_volumes0[i]
            id1 = ids_volumes1[i]
            cent0 = MM.all_centroids[id0]
            cent1 = MM.all_centroids[id1]
            lines += [gid0, gid1]
            cols += [gid1, gid0]
            data += [keq, keq]
            s_grav = keq*gama*(cent1[2] - cent0[2])
            b[gid0] += -s_grav
            b[gid1] -= -s_grav
            s_gravf[i] = s_grav

        MM.mb.tag_set_data(sgravf_tag, MM.all_intern_faces, s_gravf)
        MM.mb.tag_set_data(sgravv_tag, MM.all_volumes, b)
        n = len(MM.all_volumes)

        Tf = sp.csc_matrix((data,(lines,cols)), shape=(n, n))
        Tf = Tf.tolil()
        d1 = np.array(Tf.sum(axis=1)).reshape(1, n)[0]*(-1)
        Tf.setdiag(d1)

        As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, self.wirebasket_numbers[0])
        As['Tf'] = Tf
        self.As = As

        if data_loaded['gravity']:
            self.b = b
        else:
            self.b = np.zeros(len(MM.all_volumes))
