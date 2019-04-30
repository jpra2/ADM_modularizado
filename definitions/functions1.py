import numpy as np
from pymoab import core, types, rng, topo_util
from scipy.sparse import linalg, csc_matrix, hstack, vstack
import time



__all__ = ['get_box', 'lu_inv2', 'lu_inv3', 'lu_inv4']



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
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
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
