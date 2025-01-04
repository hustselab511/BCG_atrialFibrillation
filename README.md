# FabosNet Guide

FabosNet model is used for Ballistocardiogram signal detection of atrial fibrillation.

## Advantages
- By converting the original BCG signal into its wavelet transform temporal-spectrogram using the Morlet wavelet transform, we can enhance the local information density and effectively extract critical imaging features.
- We present a novel model that integrates atrial fibrillation (AF)-specific characteristics from different frequency bands and time domains, allowing for effective differentiation between atrial fibrillation and sporadic arrhythmias.
- The proposed FabosNet is highly adaptable to various sampling lengths, achieving an accuracy of 95.26% on signal segments as short as 5 seconds.  This offers a promising solution for early AF screening in real-world applications.

## Environment
Before you can start using the MuSFId model, you need to install the following dependencies:
- Python 3.7+
- TensorFlow 2.0+
- NumPy 1.16+
- matplotlib==3.7.1
- numpy==1.24.2
- pandas==2.0.3
- scikit-learn==1.3.2
- scipy==1.10.1
- seaborn==0.13.0
- sklearn==0.0
- sympy==1.11.1
- tensorboard==2.12.0
- torch @ http://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp38-cp38-linux_x86_64.whl
- torchvision @ http://download.pytorch.org/whl/cu118/torchvision-0.15.1%2Bcu118-cp38-cp38-linux_x86_64.whl
- tqdm @ file:///tmp/build/80754af9/tqdm_1625563689033/work



## Quick start
1. Train signal decomposition model：
   ```bash
   python FabosNetMain.py
