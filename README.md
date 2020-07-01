# Deep Isometric Learning for Visual Recognition

This repository is an official PyTorch implementation of the ICML paper:

<b>Deep Isometric Learning for Visual Recognition</b> <br>
[Haozhi Qi](https://people.eecs.berkeley.edu/~hqi/), 
[Chong You](https://sites.google.com/view/cyou),
[Xiaolong Wang](https://xiaolonw.github.io/),
[Yi Ma](http://people.eecs.berkeley.edu/~yima/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <br>
International Conference on Machine Learning (ICML), 2020 <br>
[[Project Webpage](https://haozhiqi.github.io/ISONet)], [arXiv]

## Introduction

In this project, we argue the notion of isometry is a central guiding principle for training deep ConvNet. In particular, we take a minimalist approach and show that a vanilla deep Isometric Network (ISONet) (i.e., without BN and shortcut) can be trained and achieve suprisingly good accuracy. We also show that if combined with skip connections, such near isometric networks (i.e. R-ISONet) can achieve performances on par with the standard ResNet, even without normalization at all.

## Main results

Here we show the Top-1 Classification Accuracy on ImageNet Validation dataset:

| Methods | depth18 | depth 34 | depth 50 | depth 101
| :---: | :---: | :---: | :---: | :---:  
| ISONet | 68.10 | 70.90 | 71.20 | 71.01
| R-ISONet | 69.17 | 73.43 | 76.18 | 77.08

For more results and pretrained models, see [Model Zoo](MODEL_ZOO.md).

## Using ISONet

### Data Preparation

The ImageNet folder train/val folder should locate at ```data/ILSVRC2012/train``` and ```data/ILSVRC2012/val```, respectively.

### Installation

This codebase is developed and tested with python 3.6, PyTorch 1.4, and cuda 10.1. But any version newer than that should work.

Here we gave an example of installing ISONet using conda virtual environment:
```
git clone https://github.com/HaozhiQi/ISONet
cd ISONet
conda create -y -n isonet
conda activate isonet
# install pytorch according to https://pytorch.org/
conda install -y pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch
pip install yacs
```

### Evaluation

You can download the pre-trained models from the links in [Model Zoo](MODEL_ZOO.md).

For example, if we want to test the performance of R-ISONet 18, download it from model zoo, and use the following command:

```
# change config files if you are going to test other pre-trained models
python test.py --cfg configs/IN1k-RISO18.yaml --gpus {GPU_ID} --ckpt RISO18.pt
```

### Training

To train our model from scratch, use the following command:
```
python train.py --cfg {CONFIG_FILE} --gpus {GPU_ID} --output {OUTPUT_NAME}
```

## Citing ISONet

If you find **ISONet** or this codebase helpful in your research, please consider citing:
```
@InProceedings{qi2020deep,
  author={Qi, Haozhi and You, Chong and Wang, Xiaolong and Ma, Yi and Malik, Jitendra},
  title={Deep Isometric Learning for Visual Recognition},
  booktitle={ICML},
  year={2020}
}
```
