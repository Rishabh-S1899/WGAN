# Wasserstein Generative Adversarial Network (WGAN)

## Overview

This repository contains an implementation of the Wasserstein Generative Adversarial Network (WGAN) as proposed in the original research paper [Wasserstein GAN](https://arxiv.org/abs/1701.07875). The model is trained on the CelebA dataset to generate realistic human face images.

## Features

- Wasserstein GAN implementation
- Image generation using deep learning
- Training on CelebA dataset
- TensorBoard integration for monitoring training progress

## Prerequisites

- Anaconda or Miniconda
- Python 3.10
- CUDA-compatible GPU (recommended)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rishabh-S1899/WGAN.git
cd your-wgan-repo
```

### 2. Create Conda Environment

```bash
conda create -n wgan-env python=3.10 -y
conda activate wgan-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
The torch and torchvision versions are specified with CUDA 11.8 support. If you're using a different CUDA version, you'll need to adjust accordingly.
To install these with CUDA support, you'll need to use pip with the specific URL for CUDA-enabled torch. The installation command would look like:
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```
## Training the Model

To train the WGAN:

```bash
python train.py
```

### Configuration

You can modify the model architecture in `model.py`. Key hyperparameters can be adjusted in `train.py`.

## Monitoring Training

The project uses TensorBoard for visualizing training progress:

```bash
tensorboard --logdir=runs
```

Open the provided localhost URL in your browser to monitor:
- Generated images
- Loss curves
- Model performance metrics

## Dataset

The model is trained on the CelebA (Celebrity Faces) dataset. 
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Contains over 200K celebrity images
- Ideal for face generation tasks

## Model Architecture

The WGAN architecture consists of:
- Generator Network
- Critic Network (replacing traditional discriminator)
- Wasserstein loss function
- Gradient penalty for training stability

## Requirements

- torch
- torchvision
- numpy
- tensorboard
- matplotlib

## Troubleshooting

- Ensure CUDA is properly installed if using GPU
- Check tensor dimensions match in custom layers
- Verify dataset path and format

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- [Wasserstein GAN Paper](https://arxiv.org/abs/1701.07875)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
