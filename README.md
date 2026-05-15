# MindSA
Official Implementation of MindSA in PyTorch

## Brain2Text

1) Download NSD

2) Install StableDiffusionReconstruction

3) Extract CLIP representations by running fmri_process/feature_extract_nsd.py

4) Preprocess NSD by running fmri_process/NSD_handle.py

5) Train the Reconstruction model by running brain2text_train_nsd.py

6) Reconstructing text by running brain2text_infer_nsd.py

* Our codebase builds on StableDiffusionReconstruction, MAE, SMALLCAP, stable-diffusion repositories. We would like to thank the authors.

## Citation
```
@InProceedings{Chen_2025_ICCV,
    author    = {Chen, Jiaxuan and Qi, Yu and Wang, Yueming and Pan, Gang},
    title     = {Bridging the Gap between Brain and Machine in Interpreting Visual Semantics: Towards Self-adaptive Brain-to-Text Decoding},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {21938-21948}
}
