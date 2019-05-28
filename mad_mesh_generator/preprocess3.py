# -*- coding: utf-8 -*-

from .generator1 import *
from .generator2 import Generator2
from .solving import SolvingAdm
import pdb
import os



__all__ = []

Generator2(mesh, SOL_ADM_fina, data_loaded)
SolvingAdm(mesh)
ext_h5m_out = data_loaded['input_file'] + '_solucao_adm_mono.h5m'
ext_vtk_out = data_loaded['input_file'] + '_solucao_adm_mono.vtk'
mesh.mb.write_file(ext_h5m_out)
mesh.mb.write_file(ext_vtk_out, [mesh.vv])
