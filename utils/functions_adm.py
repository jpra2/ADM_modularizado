import scipy.sparse as sp
from pymoab import types, rng, topo_util
import numpy as np

def get_OP1_ADM(mesh, n1, L1_ID_tag, L3_ID_tag, vertices):
    # vertices = mesh.wirebasket_elems[0][3]
    # L1_ID_tag = mesh.tags['l1_ID']
    # L3_ID_tag = mesh.tags['NIVEL_ID']

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

def get_OR1_ADM(mesh, n1):
    L1_ID_tag = mesh.tags['l1_ID']
    l1 = mesh.mb.tag_get_data(L1_ID_tag, mesh.all_volumes, flat=True)
    c1 = mesh.mb.tag_get_data(mesh.tags['ID_reord_tag'], mesh.all_volumes, flat=True)
    d1 = np.ones(len(l1))
    OR1_ADM = sp.csc_matrix((d1, (l1, c1)), shape=(n1, len(mesh.all_volumes)))
    return OR1_ADM

def get_OR2_ADM(mesh, n1, n2):
    L2_ID_tag = mesh.tags['l2_ID']
    L1_ID_tag = mesh.tags['l1_ID']
    l2 = mesh.mb.tag_get_data(L2_ID_tag, mesh.all_volumes, flat=True)
    c2 = mesh.mb.tag_get_data(L1_ID_tag, mesh.all_volumes, flat=True)
    d2 = np.ones(len(l2))
    OR_ADM_2 = sp.csc_matrix((d2,(l2,c2)),shape=(n2,n1))
    return OR_ADM_2

def get_OP2_ADM(mesh, n1, n2, vertices):

    OP3 = mesh.matrices['OP2_AMS'].copy()
    L3_ID_tag = mesh.tags['NIVEL_ID']
    L2_ID_tag = mesh.tags['l2_ID']
    L1_ID_tag = mesh.tags['l1_ID']

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

def get_solution_adm(mesh, Tf2, b2):

    TADM1 = mesh.matrices['OR1_ADM']*Tf2*mesh.matrices['OP1_ADM']
    TADM2 = mesh.matrices['OR2_ADM']*TADM1*mesh.matrices['OP2_ADM']
    bADM1 = mesh.matrices['OR1_ADM']*b2
    bADM2 = mesh.matrices['OR2_ADM']*bADM1
    PMS = sp.linalg.spsolve(TADM2, bADM2)
    Pms = mesh.matrices['OP2_ADM']*PMS
    Pms = mesh.matrices['OP1_ADM']*Pms

    return Pms
