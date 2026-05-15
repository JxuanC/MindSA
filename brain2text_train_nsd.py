import pandas as pd
import numpy as np
import os
import argparse
import h5py

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"#
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, GPT2Tokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from modules.brain2text import Brain2Text, Brain2TextConfig
from modules.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from utils import *
from modules.fmriencoder import fMRIViTEncoderConfig, fMRIViTEncoder
from fmri_process.dataset_augment import get_visual_text_nsd_dataset
# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
ENCODERPARAMS = {'ViT-4-4': 4, 'ViT-8-8': 8, 'ViT-16-16': 16, 'ViT-24-24': 24}

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def get_model_and_auxiliaries(args):

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("Brain2Text", Brain2TextConfig)
    AutoModel.register(Brain2TextConfig, Brain2Text)

    AutoConfig.register("fMRIViTEncoder", fMRIViTEncoderConfig)
    AutoModel.register(fMRIViTEncoderConfig, fMRIViTEncoder)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    #feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    #tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    dataset, dataloader, fmri_dim = get_visual_text_nsd_dataset(args.batch_size, subject=args.sub, tokenizer = tokenizer)

    encoder_config = fMRIViTEncoderConfig(fmri_dim, 4, 768, ENCODERPARAMS[args.encoder_cog], 
                                          ENCODERPARAMS[args.encoder_cog]) #default 16 16
    model = Brain2Text.from_encoder_decoder_pretrained(fMRIViTEncoder(encoder_config), args.decoder_name, 
                                                       cross_attention_reduce_factor = cross_attention_reduce_factor)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.max_length = CAPTION_LENGTH   
    model.config.rag = not args.disable_rag

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder   
    else:
        model.config.k = 0
  
    # freeze parameters
    for param in model.encoder.parameters():
        param.requires_grad = True

    if not args.train_decoder:
        for name, param in model.decoder.named_parameters():
            if 'crossattention' not in name:
                param.requires_grad = False

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {}M trainable parameters.'.format(num_trainable_params/100/10000))

    return model, tokenizer, dataset


def main(args):
    model, tokenizer, dataset = get_model_and_auxiliaries(args)
    
    output_dir = '{}_{}M_{}S_{}'.format('mindsa', args.attention_size, args.split, args.decoder_name)

    output_dir = os.path.join(args.experiments_dir, args.encoder_cog, args.dataset, args.sub, args.ROI, output_dir)
    
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs, 
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=False,
        save_strategy="epoch",
        save_total_limit=30, 
        logging_strategy="steps", 
        output_dir=output_dir, 
        overwrite_output_dir=True, 
        weight_decay=1e-4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--experiments_dir", type=str, default="./experiments/MindSA", help="Directory where trained models will be saved")
    #parser.add_argument("--encoder_name", type=str, default="fMRIEncoder", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--encoder_cog", type=str, default="ViT-16-16", help="Encoder parameters")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=1.75, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False, help="Whether to train the decoder in addition to the attention")
    parser.add_argument("--disable_rag", action="store_true", default=True, help="Disable retrieval augmentation")
    parser.add_argument("--n_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--sub", type=str, default='subj01')
    parser.add_argument("--dataset", type=str, default='NSD')
    parser.add_argument("--ROI", type=str, default='HVC')
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    main(args)
