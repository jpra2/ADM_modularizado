import os
import shutil
import pdb
import numpy as np
import time

def remover():

    list2 = ['OP_AMS.npz', 'OR_AMS.npz', 'residuo.npy', 'SOL_ADM_fina.npy', 'saida.csv',
            'faces_adjs_by_dual.npy', 'intern_adjs_by_dual.npy']

    for name in list2:
        try:
            # shutil.rmtree('OP_AMS.npz')
            os.remove(name)
        except:
            pass

remover()

# ks = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
ks = np.array([1.0])
np.save('ks', ks)
np.save('loop', np.array([0]))
np.save('len_ks', np.array([len(ks)]))
with open('saida.csv', 'w') as f:
    f.write('Percentual_volumes_ativos,normaL2_max,normaLinf_max,loop,kkk\n')

# from ADM_02 import *
# from ADM_02_seg import *

# from ADM_02_backup import *

for i in range(len(ks)):
    time.sleep(2)
    # from ADM_02_backup_seg import *
    # from ADM_02_seg import *
    # os.system('python ADM_02_backup_seg.py')
    os.system('python ADM_02_seg.py')
