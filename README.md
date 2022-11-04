# T5-Reaction-Prediction
This repo uses Pytorch Lightning and HuggingFace Transformer to fine-tune T5 for Chemical Reaction Prediciton.
See https://github.com/pschwllr/MolecularTransformer for training script and model training a Transformer from scratch using OpenNMT
and https://github.com/blender-nlp/MolT5 for details on pre-training T5. 

- fine_tune_pl.py trains molT5
- test_beam_search.py evaluates model checkpoint using a beam search decoder
- test_top_p.py evaluates the model checkpoint using a top-p decoder. 

- pre_training_w_mlm contains a script for pre-training T5 on a new corpus, for example if one desires to train only on SMILES, rather than a mixed corpus which molT5 used
