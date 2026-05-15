import faiss
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from fmri_process.dataset_augment import *
from transformers import CLIPFeatureExtractor, CLIPVisionModel,CLIPTextModel,CLIPTokenizer
import pickle

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

def create_visual_faiss_index(img_caps, batch_size, index_type, encoder_name, save_name):
    last_hidden_state = []
    class_embedding = []
    h5py_file = h5py.File('data/features/{}.hdf5'.format(save_name), 'w')
    feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
    #tokenizer = CLIPTokenizer.from_pretrained(encoder_name)
    clip_v_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(DEVICE)
    #clip_t_encoder = CLIPTextModel.from_pretrained(encoder_name).to(DEVICE)
    for key, value in img_caps.items():
        #imgids = images_dir[idx:idx + batch_size]
        #images = [Image.open(file_name).convert("RGB") for file_name in imgids]
        with torch.no_grad():
            pixel_values = feature_extractor(value['image'], return_tensors='pt').pixel_values.to(DEVICE)
            encodings = clip_v_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy()
            #last_hidden_state.append(encodings)
            class_embedding.append(encodings[:, 0, :])
            h5py_file.create_dataset(str(key), (50, 768), data = encodings[0])
        #if(idx>3): break
            print(key)

    class_embedding = np.vstack(class_embedding)
    embedding_dimension = class_embedding.shape[1]
    embedding_nums = class_embedding.shape[0]

    index_type = 'dot'
    if index_type == "L2":
        cpu_index = faiss.IndexFlatL2(embedding_dimension)  # 
        #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    if index_type == "dot":
        cpu_index = faiss.IndexFlatIP(embedding_dimension)  # 
        #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    if index_type == "cosine":
        # cosine = normalize & dot
        faiss.normalize_L2(class_embedding)
        cpu_index = faiss.IndexFlatIP(embedding_dimension)  # 
        #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    print(cpu_index.is_trained)
    cpu_index.add(class_embedding)  
    faiss.write_index(cpu_index, f"data/features/CLIP_{save_name}_index")
    #faiss.write_index(gpu_index, f"database/GPU_{save_name}")
    
encoder_name = 'openai/clip-vit-base-patch32'
#imgs_dir = np.concatenate([np.array(glob.glob(f"{config.ImageNet_Test}/*/*.JPEG"))])
with open('./data/nsd/subj01_ave_img_prompts.pkl', 'rb') as pickle_file:
     each_img_prompts = pickle.load(pickle_file)
create_visual_faiss_index(each_img_prompts, 128, "dot", encoder_name, 'nsd_stimuli')