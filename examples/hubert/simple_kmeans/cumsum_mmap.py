from time import time
import numpy as np
import os
import shutil
import sys

k = int(sys.argv[1])

partition = []
tot_len = 172812419
step = tot_len // 16

for i in range(0, tot_len-step, step):
    partition.append([i, min(i+step, tot_len)])
partition[-1][1] = tot_len

rootDir, subsetList, _ = next(os.walk("/gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/feats_v05/train"))

cumsum = np.zeros((partition[k][1]-partition[k][0], 768), dtype=np.float32)

for subdir in sorted(subsetList):
    if subdir == 'feat_x':
        continue
    tmp = np.load(os.path.join(rootDir, subdir, 'train_0_1.npy'), mmap_mode='r')
    cumsum += tmp[partition[k][0]:partition[k][1], :]
    #print(subdir)

mean = cumsum / (len(subsetList)-1)

np.save(os.path.join(rootDir, 'feat_x', f'train_{k}_{len(partition)}.npy'), mean)