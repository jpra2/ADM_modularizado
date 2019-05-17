import numpy as np
# from pymoab import core, types, rng, topo_util
import time
import os
# import yaml
# import sys
# from utils import pymoab_utils as utpy
# from definitions.functions1 import *
# from utils.prolongation import *
# from utils.others_utils import OtherUtils as oth
import sympy
from scipy.sparse import csc_matrix, vstack, find, linalg
import scipy
# from mad_meshv2 import MadMesh1

__all__ = ['malha_adm']

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
mono_dir = os.path.join(flying_dir, 'monofasico')
bif_dir = os.path.join(flying_dir, 'bifasico')
