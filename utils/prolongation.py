
def solve_block_matrix(topology,pos_0):
    lgp=[]
    cgp=[]
    dgp=[]
    c0=0

    st=0
    ts=0
    ta=0
    tc=0

    fl=[]
    fc=[]
    fd=[]
    for cont in range(int(len(topology)/6)):
        t1=time.time()
        Gids=topology[6*cont]
        all_faces_topo=topology[6*cont+1]
        ADJs1=topology[6*cont+2]
        ADJs2=topology[6*cont+3]
        if pos_0 > 0:
            adjsg1=topology[6*cont+4]
            adjsg2=topology[6*cont+5]
            inds1=np.where(adjsg1<pos_0)[0]
            inds2=np.where(adjsg2<pos_0)[0]
            inds_elim=np.unique(np.concatenate([inds1,inds2]))
            all_faces_topo=np.delete(all_faces_topo,inds_elim)
            ADJs1=np.delete(ADJs1,inds_elim)
            ADJs2=np.delete(ADJs2,inds_elim)
        ks_all=np.array(M1.mb.tag_get_data(M1.k_eq_tag,np.array(all_faces_topo),flat=True))
        ts+=time.time()-t1
        t2=time.time()
        int1=np.where(ADJs1<len(Gids))
        int2=np.where(ADJs2<len(Gids))
        pos_int_i=np.intersect1d(int1,int2)
        pos_int_e1=np.setdiff1d(int1,pos_int_i)
        pos_int_e2=np.setdiff1d(int2,pos_int_i)

        Lid_1=ADJs1[pos_int_i]
        Lid_2=ADJs2[pos_int_i]
        ks=ks_all[pos_int_i]

        lines1=[]
        cols1=[]
        data1=[]

        lines1.append(Lid_1)
        cols1.append(Lid_2)
        data1.append(ks)

        lines1.append(Lid_2)
        cols1.append(Lid_1)
        data1.append(ks)

        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)

        Lid_1=ADJs1[pos_int_e1]
        ks=ks_all[pos_int_e1]
        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        Lid_2=ADJs2[pos_int_e2]
        ks=ks_all[pos_int_e2]
        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)


        lines1=np.concatenate(np.array(lines1))
        cols1=np.concatenate(np.array(cols1))
        data1=np.concatenate(np.array(data1))
        M_local=csc_matrix((data1,(lines1,cols1)),shape=(len(Gids),len(Gids)))
        ta+=time.time()-t2
        tinvert=time.time()
        try:
            inv_local=lu_inv2(M_local)
        except:
            import pdb; pdb.set_trace()

        st+=time.time()-tinvert

        t3=time.time()
        ml=find(inv_local)
        fl.append(ml[0]+c0)
        fc.append(ml[1]+c0)
        fd.append(ml[2])
        lgp.append(Gids-pos_0)
        tc+=time.time()-t3
        c0+=len(Gids)

    fl=np.concatenate(np.array(fl))
    fc=np.concatenate(np.array(fc))
    fd=np.concatenate(np.array(fd))

    m_loc=csc_matrix((fd,(fl,fc)),shape=(ni,ni))
    lgp=np.concatenate(np.array(lgp))
    cgp=range(ni)
    dgp=np.ones(len(lgp))
    permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(ni,ni))
    invMatrix=permut_g*m_loc*permut_g.transpose()

    return(invMatrix)
