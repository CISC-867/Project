# Reproducing Unsupervised Synthesis of Audio via Exemplar Encoders

This repository is the official implementation of our Reproducibility Paper for CISC 867

## Requirements

- pytorch
- librosa
- webrtcvad

## Dataset downloading

See `main.ipynb` for fetching WaveNet repository and VCTK datasets.

## Training

Pre-training performed with `train.ipynb` and `afktrain.py`.  
Targeted training performed with `train-targeted.ipynb` and `afktrain-targeted.py`.

## Inference

See `wavgen.ipynb` for an example of using the model to output wav files.

## Pre-trained models

Pre-trained models are available using the links below.  
These may be deleted at some point in the future.  
The model currently does not sound like Tom Scott, and will just reproduce the input audio at best.

https://cisc867.blob.core.windows.net/checkpoints/tom/model306255_vocoder.pth

https://cisc867.blob.core.windows.net/checkpoints/tom/model306255.pth
