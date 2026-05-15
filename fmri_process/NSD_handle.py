NSD_PATH = './StableDiffusionReconstruction/'
MRIFEAT = 'mrifeat/'
STIFEAT = 'nsdfeat/subjfeat/'
import os
import numpy as np
from tqdm import tqdm
from nsda import NSDAccess
import pickle
import scipy.io

nsda = NSDAccess(os.path.join(NSD_PATH, 'nsd/'))
roi = ['ventral']
subject='subj01'
target='c'
mridir = os.path.join(NSD_PATH, MRIFEAT, subject)  
featdir = os.path.join(NSD_PATH, STIFEAT)

stims = np.load(f'{mridir}/{subject}_stims.npy')
nsd_expdesign = scipy.io.loadmat(os.path.join(NSD_PATH,'nsd/nsddata/experiments/nsd/nsd_expdesign.mat'))
sharedix = nsd_expdesign['sharedix'] -1 
feats = []
tr_idx = np.zeros(len(stims))
for idx, s in tqdm(enumerate(stims), total=len(stims)):
    if s in sharedix:
        tr_idx[idx] = 0 #test
    else:
        tr_idx[idx] = 1 #train 
    #img = nsda.read_images(s)
    #coco_info = nsda.read_image_coco_info([s],info_type='captions')
    #prompts = [p['caption'] for p in coco_info]
    feats.append(s)

np.save(f'./data/nsd/{subject}_each_stims_tridx.npy',tr_idx)
np.save(f'./data/nsd/{subject}_each_stims_idx.npy', feats)


stims = np.load(f'{mridir}/{subject}_stims_ave.npy')
#nsd_expdesign = scipy.io.loadmat(os.path.join(NSD_PATH,'nsd/nsddata/experiments/nsd/nsd_expdesign.mat'))
#sharedix = nsd_expdesign['sharedix'] -1 
feats = []
info = {}
tr_idx = np.zeros(len(stims))
for idx, s in tqdm(enumerate(stims), total=len(stims)):
    if s in sharedix:
        tr_idx[idx] = 0 #test
    else:
        tr_idx[idx] = 1 #train 
    info[s] = {}
    info[s]['image'] = nsda.read_images(s)
    coco_info = nsda.read_image_coco_info([s],info_type='captions')
    info[s]['captions'] = [p['caption'] for p in coco_info]
    feats.append(s)

np.save(f'./data/nsd/{subject}_ave_stims_tridx.npy',tr_idx)
np.save(f'./data/nsd/{subject}_ave_stims_idx.npy', feats)

file_path = f'./data/nsd/{subject}_ave_img_prompts.pkl'  # 设置保存文件的路径和名称
with open(file_path, 'wb') as yaml_file:
     pickle.dump(info, yaml_file)

