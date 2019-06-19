import os
import shutil
import pdb
import numpy as np
import time

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

    try:
        # shutil.rmtree('SOL_ADM_fina.npy')
        os.remove('saida.csv')
    except:
        pass

remover()

ks = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
np.save('ks', ks)
np.save('loop', np.array([0]))
np.save('len_ks', np.array([len(ks)]))
with open('saida.csv', 'w') as f:
    f.write('Percentual_volumes_ativos,normaL2_max,normaLinf_max,loop\n')




# from ADM_02 import *
# from ADM_02_seg import *

# from ADM_02_backup import *

for i in range(len(ks)):
    time.sleep(3)
    # from ADM_02_backup_seg import *
    # from ADM_02_seg import *
    os.system('python ADM_02_seg.py')
    import pdb; pdb.set_trace()
