import pandas as pd
import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, GPT2Tokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from modules.vit import ViT
from utils import prep_strings, postprocess_preds
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers.generation_utils import GenerationMixin
import faiss
from fmri_process.dataset_augment import get_test_visual_text_nsd_dataset
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25
def evaluate_norag_model(args, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size
    for idx in tqdm(range(0, len(eval_df[0]), bs)):
        fMRI = torch.tensor(eval_df[0][idx:idx + bs], dtype = torch.float32).to(args.device)
        image_ids = eval_df[1][idx:idx + bs]
        #encoder_labels = clip_feature[idx:idx + bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test = True) for _ in range(len(image_ids))] 
                
        with torch.no_grad():
            encoder_last_hidden_state = torch.FloatTensor(model.encoder(fMRI).cpu()).to(args.device)
            encoder_outputs = BaseModelOutput(last_hidden_state = encoder_last_hidden_state)
            preds = model.generate(encoder_outputs = encoder_outputs, decoder_input_ids = torch.tensor(decoder_input_ids).to(args.device),
                               **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)
 
        for image_id, pred in zip(image_ids, preds):
            pred = postprocess_preds(pred, tokenizer)
            out.append({"image_id": str(image_id), "caption": pred, 
                        "label1": eval_df[-1][image_id]['captions'][0],
                        "label2": eval_df[-1][image_id]['captions'][1],
                        "label3": eval_df[-1][image_id]['captions'][2],
                        "label4": eval_df[-1][image_id]['captions'][3],
                        "label5": eval_df[-1][image_id]['captions'][4]})

    return out

def evaluate_rag_model(args, tokenizer, model, eval_df, cap_labels):
    """RAG models can only be evaluated with a batch of length 1."""
    
    template = open(args.template_path).read().strip() + ' '

    out = []
    for idx in tqdm(range(len(eval_df[0]))):
        fMRI = torch.tensor(eval_df[0][idx][None,:,:], dtype = torch.float32).to(args.device)
        image_id = eval_df[1][idx]
        retrieved_caps = eval_df[2][idx]
        #image = Image.open(args.images_dir + file_name).convert("RGB")
        #pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=retrieved_caps,
                                                 k=int(args.k), is_test=True)
        with torch.no_grad():
            encoder_last_hidden_state = torch.FloatTensor(model.encoder(fMRI).cpu()).to(args.device)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state)
            pred = model.generate(encoder_outputs=encoder_outputs,
                            decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                            **args.generation_kwargs)
        
        pred = tokenizer.decode(pred[0])
        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": image_id, "retrieved_caps": retrieved_caps, "pre_cap": pred, "label": cap_labels[image_id]})

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from modules.brain2text import Brain2Text, Brain2TextConfig
    from modules.fmriencoder import fMRIViTEncoder, fMRIViTEncoderConfig
    from modules.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("fMRIViTEncoder", fMRIViTEncoderConfig)
    AutoModel.register(fMRIViTEncoderConfig, fMRIViTEncoder)

    AutoConfig.register("Brain2Text", Brain2TextConfig)
    AutoModel.register(Brain2TextConfig, Brain2Text)

@torch.no_grad()
def rag_captions(data, sortedimageIDs, caps, args):
    fmri_dim = data[0].shape[-1]
    roi_num = data[0].shape[-2]
    retrieval_model = ViT(fmri_dim, roi_num, 512)
    retrieval_model.load_state_dict(torch.load(args.retrieval_model_path, map_location = 'cpu'))
    retrieval_model.to(args.device).eval()

    retrieval_index = faiss.read_index(args.retrieval_index_path)
    res = faiss.StandardGpuResources()  
    retrieval_index = faiss.index_cpu_to_gpu(res, 0, retrieval_index)

    fmri_embedding = retrieval_model(torch.tensor(data[0], dtype = torch.float32).to(args.device))[:, 0, :]
    fmri_embedding = fmri_embedding / fmri_embedding.norm(dim=-1, keepdim=True)
    dis, nns = retrieval_index.search(fmri_embedding.cpu().numpy().astype(np.float32), args.k) 

    return [data[0], data[1], [[caps[str(sortedimageIDs[nns[n][k]]).split('/')[-1]] for k in range(args.k)] for n in range(nns.shape[0])]]


def main(args):

    register_model_and_config()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    X_te, te_id, Y_te, Caps_te = get_test_visual_text_nsd_dataset()
    data = [X_te, te_id, Y_te, Caps_te]
    
    args.outfile_name = '{}_preds.json'.format(args.infer_data)

    if args.disable_rag:
        output_dir =  'mindsa_{}M_{}S_gpt2'.format(args.attention_size, args.split)
        args.k = 0
        infer_fn = evaluate_norag_model


    args.model_path = os.path.join(args.model_path, args.encoder_cog, args.dataset, args.sub, args.ROI, output_dir)

    # load and configure tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id, 'bos_token_id': tokenizer.bos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, tokenizer, checkpoint_path, data, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, tokenizer, checkpoint_path, data, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--model_path", type=str, default='./experiments/MindSA', help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")
    parser.add_argument("--infer_data", type=str, default='test', help="Use test data or train data")
    parser.add_argument("--encoder_cog", type=str, default="ViT-16-16", help="Encoder parameters")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--disable_rag", action="store_true", default=True, help="Disable retrieval augmentation or not")
    parser.add_argument("--template_path", type=str, default="data/template.txt", help="TXT file with template")
    parser.add_argument("--attention_size", type=float, default=1.75, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size; only matter if evaluating a norag model")
    parser.add_argument("--sub", type=str, default='subj01', help="Whether to blank visual features")
    parser.add_argument("--dataset", type=str, default='NSD')
    parser.add_argument("--ROI", type=str, default='HVC')
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    main(args)
   
