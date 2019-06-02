import os
import shutil
import pdb


def remover():

    try:
        # shutil.rmtree('OP_AMS.npz')
        os.remove('OP_AMS.npz')
    except:
        pass
    try:
        # shutil.rmtree('OP_AMS.npz')
        os.remove('OR_AMS.npz')
    except:
        pass
    try:
        # shutil.rmtree('residuo.npy')
        os.remove('residuo.npy')
    except:
        pass

    try:
        # shutil.rmtree('SOL_ADM_fina.npy')
        os.remove('SOL_ADM_fina.npy')
    except:
        pass

remover()

# from ADM_02 import *
from ADM_02_seg import *
