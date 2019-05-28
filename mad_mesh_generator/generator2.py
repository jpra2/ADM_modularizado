from pymoab import types, rng, topo_util
import numpy as np
import scipy.sparse as sp
import cProfile
import pstats
import io


class Generator2:

    def __init__(self, mesh, SOL_ADM_fina, data_loaded):

        # pr = cProfile.Profile()
        # pr.enable()
        gam2 = GenerateAdmMesh2(mesh)
        gam2.definir_parametros(mesh)
        gam2.set_erro_aprox(mesh, SOL_ADM_fina)
        mesh.n1, mesh.n2 = gam2.generate_adm_mesh(mesh, SOL_ADM_fina)
        #########################
        # pr.disable()
        # pr.print_stats()
        ###########################
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        ext_h5m_out = data_loaded['input_file'] + '_malha_adm.h5m'
        ext_vtk_out = data_loaded['input_file'] + '_malha_adm.vtk'
        mesh.mb.write_file(ext_h5m_out)
        mesh.mb.write_file(ext_vtk_out, [mesh.vv])


class GenerateAdmMesh2:

    def __init__(self, mesh):

        self.ares2_tag = mesh.mb.tag_get_handle("ares_2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        mesh.tags['ares_2'] = self.ares2_tag
        self.ares4_tag=mesh.mb.tag_get_handle("ares_4", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        mesh.tags['ares4'] = self.ares4_tag
        self.ares9_tag=mesh.mb.tag_get_handle("ares_9", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        mesh.tags['ares9'] = self.ares9_tag
        self.ares6_tag=mesh.mb.tag_get_handle("ares_6", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        mesh.tags['ares6'] = self.ares6_tag

    def definir_parametros(self, mesh):

        ares2_tag = self.ares2_tag

        D1_tag = mesh.tags['d1']

        meshset_by_L2 = mesh.mb.get_child_meshsets(mesh.L2_meshset)
        self.meshset_by_L2 = meshset_by_L2
        max1=0
        for m2 in meshset_by_L2:
            meshset_by_L1=mesh.mb.get_child_meshsets(m2)
            for m1 in meshset_by_L1:
                ver_1=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
                viz_vert=mesh.mtu.get_bridge_adjacencies(ver_1,1,3)
                k_vert=mesh.mb.tag_get_data(mesh.tags['PERM'],ver_1)[:,0]
                facs_ver1=mesh.mtu.get_bridge_adjacencies(ver_1,2,2)
                max_r=0
                vers=[]
                somak=0
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

                            if len(ares)>max1:
                                print(len(ares),len(novas_ares))
                                max1=len(ares)
                            if len(novas_ares)==0:
                                break
                        a1=ares
                        gids_ares=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],a1,flat=True)
                        facs_ares=mesh.mtu.get_bridge_adjacencies(a1,2,2)
                        somak+= sum(mesh.mb.tag_get_data(mesh.tags['K_EQ'],facs_ares,flat=True))/k_vert
                        ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                        ares_m=mesh.mb.create_meshset()
                        mesh.mb.add_entities(ares_m,ares)
                        verts=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
                        ares_ares=verts=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                        v_verts=mesh.mtu.get_bridge_adjacencies(verts,2,3)
                        ares_ares=rng.unite(ares_ares,verts)
                        ares=rng.unite(ares,v_verts)
                        k_ares_max=mesh.mb.tag_get_data(mesh.tags['PERM'],ares)[:,0].max()
                        k_ares_min=mesh.mb.tag_get_data(mesh.tags['PERM'],ares)[:,0].min()
                        k_ares_med=sum(mesh.mb.tag_get_data(mesh.tags['PERM'],a1)[:,0])/len(ares)
                        ver_2=np.uint64(rng.subtract(verts,ver_1))
                        k_ver2=mesh.mb.tag_get_data(mesh.tags['PERM'],ver_2)[0][0]
                        vers.append(ver_2)
                        perm_ares=mesh.mb.tag_get_data(mesh.tags['PERM'],ares)[:,0]
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
                perm_viz=mesh.mb.tag_get_data(mesh.tags['PERM'],viz_vert)[:,0]
                raz=float(k_vert/perm_viz.min())
                # mesh.mb.tag_set_data(raz_tag, ver_1,raz)
                #mesh.mb.tag_set_data(var_tag, ver_1,var)
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

    def set_erro_aprox(self, mesh, SOL_ADM_f):

        ni = mesh.wirebasket_numbers[0][0]
        ares4_tag = self.ares4_tag
        ares6_tag = self.ares6_tag
        ares9_tag = self.ares9_tag
        D1_tag = mesh.tags['d1']
        gids_vert=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],mesh.wirebasket_elems[0][3],flat=True)
        T = mesh.matrices['Tf']
        vertices = mesh.wirebasket_elems[0][3]

        sol_vers=sp.csc_matrix(SOL_ADM_f[gids_vert]).transpose()
        k1 = np.mean(SOL_ADM_f)
        prol=mesh.matrices['OP1_AMS']
        rest=mesh.matrices['OR1_AMS']
        sol_prol=(prol*sol_vers).transpose().toarray()[0]
        dif=abs((sol_prol-SOL_ADM_f)/np.repeat(k1,len(mesh.all_volumes)))
        GIDs=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],mesh.all_volumes,flat=True)
        mesh.mb.tag_set_data(ares4_tag,mesh.all_volumes,dif[GIDs])
        try:
            RTP=rest*T*prol
            som_col=np.array(prol.sum(axis=0))[0]
            diag_RTP=abs(RTP[range(len(vertices)),range(len(vertices))].toarray()[0])
            mesh.mb.tag_set_data(ares6_tag,vertices,diag_RTP)
        except:
            import pdb; pdb.set_trace()

        sol_vers=sol_vers.tocsr()
        ident=sp.identity(len(vertices)).tocsr()
        ident2=ident.copy()
        delta=1
        cont=0

        for v in vertices:
            ident2[cont,cont]+=delta
            s_v=ident2*sol_vers
            s_p=(prol*s_v).transpose().toarray()[0]
            d1=((s_p-sol_prol)/sol_prol)[range(ni)]
            d1=d1.max()
            mesh.mb.tag_set_data(ares9_tag,v,d1)
            ident2=ident.copy()
            cont+=1

        for m2 in self.meshset_by_L2:
            tem_poço_no_vizinho=False
            meshset_by_L1=mesh.mb.get_child_meshsets(m2)
            for m1 in meshset_by_L1:
                elem_by_L1 = mesh.mb.get_entities_by_handle(m1)
                ver_1=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))

                ar6=float(mesh.mb.tag_get_data(ares6_tag,ver_1))
                mesh.mb.tag_set_data(ares6_tag,elem_by_L1,np.repeat(ar6,len(elem_by_L1)))

    def generate_adm_mesh(self, mesh, SOL_ADM_f):

        n1=0
        n2=0
        aux=0
        meshset_by_L2 = self.meshset_by_L2
        D1_tag = mesh.tags['d1']
        finos = list(mesh.volumes_f)
        intermediarios = mesh.intermediarios
        L1_ID_tag = mesh.tags['l1_ID']
        L2_ID_tag = mesh.tags['l2_ID']
        L3_ID_tag = mesh.tags['NIVEL_ID']
        ares2_tag = self.ares2_tag
        ares6_tag = self.ares6_tag
        ares4_tag = self.ares4_tag
        ares9_tag = self.ares9_tag

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
                    viz_vertice=mesh.mtu.get_bridge_adjacencies(ver_1,2,3)
                    k_vert=float(mesh.mb.tag_get_data(mesh.tags['PERM'],ver_1)[:,0])
                    k_viz=mesh.mb.tag_get_data(mesh.tags['PERM'],viz_vertice)[:,0]
                    # raz=float(mesh.mb.tag_get_data(raz_tag,ver_1))
                    perm1=mesh.mb.tag_get_data(mesh.tags['PERM'],elem_by_L1)
                    med1_x=sum(perm1[:,0])/len(perm1[:,0])
                    med1_y=sum(perm1[:,4])/len(perm1[:,4])
                    med1_z=sum(perm1[:,8])/len(perm1[:,8])
                    gids_primal=mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'],elem_by_L1)
                    press_primal=SOL_ADM_f[gids_primal]
                    ares=mesh.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([2]))

                    viz_ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                    ares_m=mesh.mb.create_meshset()
                    mesh.mb.add_entities(ares_m,viz_ares)
                    viz_ares_ares=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                    ares=viz_ares_ares

                    viz_ares=mesh.mtu.get_bridge_adjacencies(ares,2,3)
                    ares_m=mesh.mb.create_meshset()
                    mesh.mb.add_entities(ares_m,viz_ares)
                    viz_ares_ares=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                    viz_ares_ver=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
                    viz_ares_fac=mesh.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([1]))
                    viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_ver)
                    viz_ares_ares=rng.unite(viz_ares_ares,ver_1)
                    viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_fac)
                    ares=viz_ares_ares

                    k_ares_max=mesh.mb.tag_get_data(mesh.tags['PERM'],ares)[:,0].max()
                    k_ares_min=mesh.mb.tag_get_data(mesh.tags['PERM'],ares)[:,0].min()
                    r_k_are_ver=float((k_ares_max-k_ares_min)/k_vert)
                    #mesh.mb.tag_set_data(ares_tag, ares, np.repeat(r_k_are_ver,len(ares)))
                    r_k_are_ver=float(mesh.mb.tag_get_data(ares2_tag,ver_1))
                    # var=float(mesh.mb.tag_get_data(var_tag,ver_1))
                    # var2=float(mesh.mb.tag_get_data(var2_tag,ver_1))
                    # ar=float(mesh.mb.tag_get_data(ares_tag,ver_1))
                    # ar3=float(mesh.mb.tag_get_data(ares3_tag,ver_1))
                    ar4=mesh.mb.tag_get_data(ares4_tag,mesh.mtu.get_bridge_adjacencies(elem_by_L1,2,3),flat=True).max()
                    #ar5=float(mesh.mb.tag_get_data(ares5_tag,ver_1))
                    ar6=float(mesh.mb.tag_get_data(ares6_tag,ver_1))

                    viz_meshset=mesh.mtu.get_bridge_adjacencies(elem_by_L1,2,3)
                    so_viz=rng.subtract(viz_meshset,elem_by_L1)
                    m7=max(mesh.mb.tag_get_data(ares6_tag,so_viz,flat=True))
                    # m8=max(mesh.mb.tag_get_data(var_tag,so_viz,flat=True))
                    # a7=ar6/var
                    # a8=ar4*(1+a7)**3
                    a9=float(mesh.mb.tag_get_data(ares9_tag,ver_1))
                    # mesh.mb.tag_set_data(ares7_tag,ver_1,a7)
                    # mesh.mb.tag_set_data(ares8_tag,elem_by_L1,np.repeat(ar4/a9,len(elem_by_L1)))
                    mesh.mb.tag_set_data(ares9_tag,elem_by_L1,np.repeat(a9,len(elem_by_L1)))

                    # max_grad=get_max_grad(m1)
                    # mesh.mb.tag_set_data(grad_tag,elem_by_L1,np.repeat(max_grad/grad_p_res,len(elem_by_L1)))
                    #if med1_x<val_barreira or med1_x>val_canal or raz>raz_lim or var<1 or r_k_are_ver>100 or (max_grad>20*grad_p_res and (r_k_are_ver>200 or var2>10000)) or (max_grad>5*grad_p_res and r_k_are_ver>10000) or (max_grad>20*grad_p_res and var<15) or (max_grad>50*grad_p_res and var<25) or ar>1000:
                    #if med1_x<val_barreira or med1_x>val_canal or raz>raz_lim or (max_grad>35*grad_p_res and (r_k_are_ver>500 or var<2 or (r_k_are_ver>100 and ar>5)) or ar3>3) or (max_grad>20*grad_p_res and (r_k_are_ver>400 or ar>100 or var2>200)) or var<1 or (r_k_are_ver>5000 and var<2) or r_k_are_ver>10000:
                    #if med1_x<val_barreira or med1_x>val_canal or (ar3>20 and var<1) or (r_k_are_ver>10000 and ar4>0.01) or (ar4>0.05 and (ar3>2 or r_k_are_ver>100 or ar>20 or var<3 or med1_x<vt)) or (r_k_are_ver>2000 and var<4) or ar5<30:
                    #if (ar5<50 and ar4>0.5) or (ar5<40 and ar4>0.4) or (ar5<30 and ar4>0.3) or var<1 or ar6<20: # bom _______ or ((ar5<30 or var<5) and ar4>0.01)
                    #if ar4>0.2 or (ar6<200 and ar4>0.2) or (ar6<80 and ar4>0.05) or ((var<10 and a8>10) and ar4>0.04) or ar6<20 or var<1 or a8>100 or (a7>0.5 and ar4>0.02) or a7>1:
                    #if (ar4/a9>0.02 and (ar6<40 or a9<0.05 or r_k_are_ver>10000)) or (ar4/a9>0.04 and (ar6<40 or a9<0.1 or r_k_are_ver>5000)) or (ar4/a9>0.2 and (ar6<300 or a9<0.3 or r_k_are_ver>1000)) or (ar4/a9>0.3 and (ar6<400 or a9<0.5 or r_k_are_ver>500)):#excelente
                    #if (ar4/a9>0.02 and (ar6<40 or a9<0.05 or r_k_are_ver>10000)) or (ar4/a9>0.04 and (ar6<40 or a9<0.1 or r_k_are_ver>5000)) or (ar4/a9>0.1 and (ar6<100 and a9<0.4 or r_k_are_ver>100)) or (ar4/a9>0.2 and (ar6<300 and a9<0.3 and r_k_are_ver>1000)) or (ar4/a9>0.25 and (ar6<400 or a9<0.5 or r_k_are_ver>500)):#excelente
                    if (ar4/a9>0.02 and (ar6<40 or a9<0.05 or r_k_are_ver>10000)) or (ar4/a9>0.04 and (ar6<40 or a9<0.1 or r_k_are_ver>5000)) or (ar4/a9>0.1 and (ar6<100 and a9<0.4 and r_k_are_ver>100)) or (ar4/a9>0.2 and (ar6<300 and a9<0.3 and r_k_are_ver>1000)) or (ar4/a9>0.25 and (ar6<400 or a9<0.5 or r_k_are_ver>500)):
                        #if ar>20 or r_k_are_ver>2000:
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

                vers=mesh.mb.get_entities_by_type_and_tag(m2, types.MBHEX, np.array([D1_tag]), np.array([3]))
                r_k_are_ver=mesh.mb.tag_get_data(ares2_tag,vers,flat=True)

                if max(r_k_are_ver)>1000:
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
