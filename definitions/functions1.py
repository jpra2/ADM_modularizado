import numpy as np
from pymoab import core, types, rng, topo_util
from scipy.sparse import linalg, csc_matrix, hstack, vstack, csr_matrix
import time
from scipy.sparse import find, csc_matrix

__all__ = ['get_box', 'lu_inv2', 'lu_inv3', 'lu_inv4', 'solve_block_matrix']

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

def lu_inv2(M):
    L=M.shape[0]
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        t0=time.time()
        lc=range(L)
        d=np.repeat(1,L)
        B=csc_matrix((d,(lc,lc)),shape=(L,L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            t0=time.time()
            l=range(s*i,s*(i+1))
            B=csc_matrix((d,(l,c)),shape=(L,s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=range(s*n,L)
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(L,r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
    #print(time.time()-tinv,M.shape[0],"tempo de inversão")
    return inversa

def lu_inv3(M,lines):
    lines=np.array(lines)
    L=len(lines)
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B,'T')).transpose()
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B,'T')).transpose()
            else:
                inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))
    f=find(inversa)
    ll=f[0]
    c=f[1]
    d=f[2]
    pos_to_line=dict(zip(range(len(lines)),lines))
    lg=[pos_to_line[l] for l in ll]
    inversa=csc_matrix((d,(lg,c)),shape=(M.shape[0],M.shape[0]))
    #print(time.time()-tinv,L,"tempo de inversão")
    return inversa

def lu_inv4(M,lines):
    lines=np.array(lines)
    cols=lines
    L=len(lines)
    s=500
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        try:
            B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        except:
            import pdb; pdb.set_trace()
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

    tk1=time.time()
    #f=find(inversa.tocsr())
    #l=f[0]
    #cc=f[1]
    #d=f[2]
    #pos_to_col=dict(zip(range(len(cols)),cols))
    #cg=[pos_to_col[c] for c in cc]
    lp=range(len(cols))
    cp=cols
    dp=np.repeat(1,len(cols))
    permut=csc_matrix((dp,(lp,cp)),shape=(len(cols),M.shape[0]))
    inversa=csc_matrix(inversa*permut)

    #inversa1=csc_matrix((d,(l,cg)),shape=(M.shape[0],M.shape[0]))
    #inversa=inversa1
    print(tk1-tinv,L,time.time()-tk1,len(lines),'/',M.shape[0],"tempo de inversão")
    return inversa

def solve_block_matrix(topology, pos_0, mb, k_eq_tag, n0):
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
        ks_all=np.array(mb.tag_get_data(k_eq_tag,np.array(all_faces_topo),flat=True))
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

    m_loc=csc_matrix((fd,(fl,fc)),shape=(n0,n0))
    lgp=np.concatenate(np.array(lgp))
    cgp=range(n0)
    dgp=np.ones(len(lgp))
    permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(n0,n0))
    invMatrix=permut_g*m_loc*permut_g.transpose()

    return(invMatrix)
