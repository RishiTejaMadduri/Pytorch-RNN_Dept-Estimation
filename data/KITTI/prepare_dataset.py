#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import pickle
import argparse
from multiprocessing import Pool
from kitti_utils_1 import*


# In[1]:

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="The path to where processed dataset will be stored")
parser.add_argument("--kitti_path", type=str, required=True, help="The path to the kitti data directory")
parser.add_argument("--depth_path", type=str, required=True, help="The path to the depth_maps data directory")

print("Entering Main")

def main(args):
#Not Needed     parser.add_argument("--threads", type=int, default=16, help="Number of threads")
    print("Running Main")
    outputdir=args.output_dir
    kitti_path=args.kitti_path
    depth_path=args.depth_path 
    os.makedirs(outputdir)
#Not Needed      threads=args.threads
    
#     with open('kitti_train.txt','r') as f:
#         sequences=f.read().splitlines()

    
    lists=[]
    
#     for i, seq_name in enumerate(sequences):
#         print(seq_name)
    seq_name='2011_09_26_drive_0001_sync'
    outfile = os.path.join(outputdir, 'cam3_'+ seq_name + ".pkl")
    lists.append((outfile, kitti_path,depth_path, seq_name))

    created_groups=create_samples_from_sequence_kitti(outfile, kitti_path, depth_path, seq_name)
        
    print('created : ', created_groups)
    
    return 0


# In[ ]:



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
