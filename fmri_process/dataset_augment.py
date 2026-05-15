import os
import clip
import bdpy
import torch
import faiss
import json
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio
from modules.vit import ViT
from einops import rearrange
NSD_PATH = './StableDiffusionReconstruction/'
MRIFEAT = 'mrifeat/'
STIFEAT = 'nsdfeat/subjfeat/'
import os
import numpy as np
from fmri_process.nsda import NSDAccess
import pickle
import h5py
from sklearn.preprocessing import normalize

class Visual_Text_NSD_Dataset(Dataset):
     def __init__(self, fMRI, image, caps, tokenizer, train = True, max_caption_length = 25):
        self.tokenizer = tokenizer
        self.fMRI = fMRI
        self.image = image
        self.train = train
        self.caps = caps
        self.SIMPLE_PREFIX = "This image shows "
        self.CAPTION_LENGTH = max_caption_length
        self.template = self.SIMPLE_PREFIX
        self.max_target_length = (max_caption_length
                                + len(tokenizer.encode(self.template)))

     def __len__(self):
        return len(self.caps)

     def prep_strings(self, text, tokenizer, retrieved_caps = None): 
        if not self.train:
            padding = False
            truncation = False
        else:
            padding = True 
            truncation = True
        
        if retrieved_caps is not None:
            infix = '\n\n'.join(retrieved_caps) + '.'
            prefix = self.template.replace('||', infix)
        else:
            prefix = self.SIMPLE_PREFIX

        prefix_ids = tokenizer.encode(prefix)
        len_prefix = len(prefix_ids)

        text_ids = tokenizer.encode(text, add_special_tokens = False)
        if truncation:
            text_ids = text_ids[:self.CAPTION_LENGTH]
        input_ids = prefix_ids + text_ids if self.train else prefix_ids

        # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
        label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
        if padding:
            input_ids += [tokenizer.pad_token_id] * (self.max_target_length - len(input_ids))
            label_ids += [-100] * (self.max_target_length - len(label_ids))
        
        if not self.train:
            return input_ids
        else:  
            return input_ids, label_ids
     
     def __getitem__(self, idx):
        visual_features = self.image[idx]
        cap = self.caps[idx]
        fmri = self.fMRI[idx]

        if(self.train):
            selected_no = random.randint(0, len(cap)-1)         
            cap = cap[selected_no]
        decoder_input_ids, labels = self.prep_strings(cap, self.tokenizer)
        data = {'encoder_inputs': fmri.astype(np.float32), 'encoder_labels': visual_features, 
                'decoder_input_ids': np.array(decoder_input_ids), 'decoder_labels': np.array(labels)}
        return data

def get_visual_text_nsd_dataset(batch_size, subject='subj01', rois=['ventral'], tokenizer = None):
    nsda = NSDAccess(os.path.join(NSD_PATH, 'nsd/'))
    target='c'
    mridir = os.path.join(NSD_PATH, MRIFEAT, subject)  
    featdir = os.path.join(NSD_PATH, STIFEAT)

    each_tridx = np.load(f'./data/nsd/{subject}_each_stims_tridx.npy')
    ave_tridx = np.load(f'./data/nsd/{subject}_ave_stims_tridx.npy')
    each_idx = np.load(f'./data/nsd/{subject}_each_stims_idx.npy')
    with open(f'./data/nsd/{subject}_ave_img_prompts.pkl', 'rb') as pickle_ave:
        ave_img_prompts = pickle.load(pickle_ave)
    file_path = './data/features/nsd_stimuli.hdf5'  
    CLIP_FEATURES = h5py.File(file_path, 'r')

    tr_id = np.array(each_idx)[each_tridx == 1]
    te_id = np.array(list(ave_img_prompts.keys()))[ave_tridx == 0]
    X = []
    X_te = []
    for croi in rois:
        cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        X.append(normalize(cX, norm='l2', axis=1))
        X_te.append(normalize(cX_te, norm='l2', axis=1))
    X = rearrange(np.hstack(X),'n (i d)-> n i d', i = 4)
    X_te = rearrange(np.hstack(X_te),'n (i d)-> n i d', i = 4)

    Y = [CLIP_FEATURES[str(id)][()] for id in tr_id]
    Caps = [ave_img_prompts[id]['captions'] for id in tr_id]
    Y_te = [CLIP_FEATURES[str(id)][()] for id in te_id]
    Caps_te = [ave_img_prompts[id]['captions'] for id in tr_id]
    
    train_dataset = Visual_Text_NSD_Dataset(X, Y, Caps, tokenizer)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    return train_dataset, train_dataloader, X.shape[-1]#, #test_dataloader, fmri_dim

def get_test_visual_text_nsd_dataset(subject='subj01', rois=['ventral']):
    nsda = NSDAccess(os.path.join(NSD_PATH, 'nsd/'))
    target='c'
    mridir = os.path.join(NSD_PATH, MRIFEAT, subject)  
    featdir = os.path.join(NSD_PATH, STIFEAT)

    ave_tridx = np.load(f'./data/nsd/{subject}_ave_stims_tridx.npy')
    with open(f'./data/nsd/{subject}_ave_img_prompts.pkl', 'rb') as pickle_ave:
        ave_img_prompts = pickle.load(pickle_ave)
    CLIP_FEATURES = h5py.File('./data/features/nsd_stimuli.hdf5', 'r')

    te_id = np.array(list(ave_img_prompts.keys()))[ave_tridx == 0]
    X_te = []
    for croi in rois:
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        X_te.append(normalize(cX_te, norm='l2', axis=1))
    X_te = rearrange(np.hstack(X_te),'n (i d)-> n i d', i = 4)

    Y_te = [CLIP_FEATURES[str(id)][()] for id in te_id]
    Caps_te = [ave_img_prompts[id]['captions'] for id in te_id]
    
    return X_te, te_id, Y_te, ave_img_prompts