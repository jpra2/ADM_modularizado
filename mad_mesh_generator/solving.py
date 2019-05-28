import numpy as np
import scipy.sparse as sp
import pdb
import os
from pymoab import types, rng, topo_util
from utils import functions_adm
from utils.others_utils import OtherUtils as oth


class SolvingAdm:

    def __init__(self, mesh):
        L1_ID_tag = mesh.tags['l1_ID']
        L2_ID_tag = mesh.tags['l2_ID']
        L3_ID_tag = mesh.tags['NIVEL_ID']
        vertices = mesh.wirebasket_elems[0][3]
        pms_tag = mesh.tags['PMS2']

        mesh.matrices['OP1_ADM'] = functions_adm.get_OP1_ADM(mesh, mesh.n1, L1_ID_tag, L3_ID_tag, vertices)
        mesh.matrices['OR1_ADM'] = functions_adm.get_OR1_ADM(mesh, mesh.n1)
        mesh.matrices['OR2_ADM'] = functions_adm.get_OR2_ADM(mesh, mesh.n1, mesh.n2)
        mesh.matrices['OP2_ADM'] = functions_adm.get_OP2_ADM(mesh, mesh.n1, mesh.n2, vertices)
        Pms = functions_adm.get_solution_adm(mesh, mesh.matrices['Tf2'], mesh.matrices['b2'])
        
        wire_elems = []
        for elems in mesh.wirebasket_elems[0]:
            wire_elems += list(elems)

        mesh.mb.tag_set_data(pms_tag, wire_elems, Pms)
        Pf = oth.get_solution(mesh.matrices['Tf2'], mesh.matrices['b2'])
        pf_tag = mesh.mb.tag_get_handle('Pf', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        mesh.tags['Pf'] = pf_tag
        mesh.mb.tag_set_data(pf_tag, wire_elems, Pf)
