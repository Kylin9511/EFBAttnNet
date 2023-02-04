## Overview
This is the PyTorch implementation of paper [Towards Efficient Subarray Hybrid Beamforming: Attention Network-based Practical Feedback in FDD Massive MU-MIMO System]() which has been submitted to IEEE for possible publication. The test script and trained models are listed here and the key results can be reproduced as a validation of our work .

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.8
- [PyTorch >= 1.10.1](https://pytorch.org/get-started/locally/)
- [einops](https://github.com/arogozhnikov/einops)
- [fvcore](https://github.com/facebookresearch/fvcore)


## Project Preparation

#### A. Data Preparation
The channel state information (CSI) matrix is generated according to the influential clustered Saleh Valenzuela (SV) model. The test dataset is provided in [Google Drive]() or [Baidu Netdisk](), which is easy for you to download and reproduce the experiment results.
You can also generate your own dataset according to the SV channel model. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading
The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Baidu Netdisk]() or [Google Drive]()

#### C. Project Tree Arrangement
We recommend you to arrange the project tree as follows.

```
home
├── EFBAttnNet  # The cloned EFBAttnNet repository
│   ├── dataset
│   ├── model
│   ├── utils
│   ├── main.py
├── K2M64Lp2L4      # Checkpoints for the scenario (K=2,M=64,Lp=2,L=4) with different B
│   ├── EFBAttnNet  # EFBAttnNet checkpoints folder
│   │     ├── model_B3.pth
│   │     ├── ...
│   ├── EFBRefineNet # EFBRefineNet checkpoints folder
│   │     ├── model_B3.pth
│   │     ├── ...
├── K2M64Lp2L8      # Checkpoints for the scenario (K=2,M=64,Lp=2,L=8) with different B
│   ├── EFBAttnNet  # EFBAttnNet checkpoints folder
│   │     ├── model_B3.pth
│   │     ├── ...
│   ├── EFBRefineNet # EFBRefineNet checkpoints folder
│   │     ├── model_B3.pth
│   │     ├── ...
├── TestData_K2M64Lp2_10000.mat  # The test dataset
├── evaluate.sh  # The test script
...
```

## Results and Reproduction

The main results of deep learning methods reported in our paper are presented in the following tables. All the listed results are marked in Fig. 4 and Fig. 5 in our paper. Our proposed EFBAttnNet utilizes advanced attention mechanism to achieve good performance with an extremely lightweight model. We also choose the end-to-end model in this [paper](https://ieeexplore.ieee.org/document/9814463) as a benchmark. Its model, which we called EFBRefineNet, is transferred into our subarray hybrid precoding scheme for fare comparison. We also provide its checkpoints in this repository.

Pilot Length L| Feedback Bits B | Model | Sum Rate (Bits/s/Hz) | Checkpoint
:--: | :--: | :--: | :--: | :--:
4 | 3 | EFBAttnNet | 4.473 | K2M64Lp2L4/EFBAttnNet/model_B3.pth
4 | 3 | EFBRefineNet | 4.600 | K2M64Lp2L4/EFBRefineNet/model_B3.pth
4 | 5 | EFBAttnNet | 4.821 | K2M64Lp2L4/EFBAttnNet/model_B5.pth
4 | 5 | EFBRefineNet | 5.192 | K2M64Lp2L4/EFBRefineNet/model_B5.pth
4 | 8 | EFBAttnNet | 6.571 | K2M64Lp2L4/EFBAttnNet/model_B8.pth
4 | 8 | EFBRefineNet | 7.578 | K2M64Lp2L4/EFBRefineNet/model_B8.pth
4 | 10 | EFBAttnNet | 8.511 | K2M64Lp2L4/EFBAttnNet/model_B10.pth
4 | 10 | EFBRefineNet | 9.286 | K2M64Lp2L4/EFBRefineNet/model_B10.pth
4 | 20 | EFBAttnNet | 9.352 | K2M64Lp2L4/EFBAttnNet/model_B20.pth
4 | 20 | EFBRefineNet | 9.656 | K2M64Lp2L4/EFBRefineNet/model_B20.pth
4 | 30 | EFBAttnNet | 9.520 | K2M64Lp2L4/EFBAttnNet/model_B30.pth
4 | 30 | EFBRefineNet | 9.732 | K2M64Lp2L4/EFBRefineNet/model_B30.pth
4 | 40 | EFBAttnNet | 9.633 | K2M64Lp2L4/EFBAttnNet/model_B40.pth
4 | 40 | EFBRefineNet | 9.757 | K2M64Lp2L4/EFBRefineNet/model_B40.pth
4 | 50 | EFBAttnNet | 9.549 | K2M64Lp2L4/EFBAttnNet/model_B50.pth
4 | 50 | EFBRefineNet | 9.778 | K2M64Lp2L4/EFBRefineNet/model_B50.pth
4 | 60 | EFBAttnNet | 9.637 | K2M64Lp2L4/EFBAttnNet/model_B60.pth
4 | 60 | EFBRefineNet | 9.713 | K2M64Lp2L4/EFBRefineNet/model_B60.pth


Pilot Length L | Feedback Bits B | Model | Sum Rate (Bits/s/Hz) | Checkpoint
:--: | :--: | :--: | :--: | :--:
8 | 3 | EFBAttnNet | 4.563 | K2M64Lp2L8/EFBAttnNet/model_B3.pth
8 | 3 | EFBRefineNet | 5.317 | K2M64Lp2L8/EFBRefineNet/model_B3.pth
8 | 5 | EFBAttnNet | 5.433 | K2M64Lp2L8/EFBAttnNet/model_B5.pth
8 | 5 | EFBRefineNet | 6.132 | K2M64Lp2L8/EFBRefineNet/model_B5.pth
8 | 8 | EFBAttnNet | 7.349 | K2M64Lp2L8/EFBAttnNet/model_B8.pth
8 | 8 | EFBRefineNet | 7.800 | K2M64Lp2L8/EFBRefineNet/model_B8.pth
8 | 10 | EFBAttnNet | 7.501 | K2M64Lp2L8/EFBAttnNet/model_B10.pth
8 | 10 | EFBRefineNet | 9.363 | K2M64Lp2L8/EFBRefineNet/model_B10.pth
8 | 20 | EFBAttnNet | 10.76 | K2M64Lp2L8/EFBAttnNet/model_B20.pth
8 | 20 | EFBRefineNet | 11.24 | K2M64Lp2L8/EFBRefineNet/model_B20.pth
8 | 30 | EFBAttnNet | 10.93 | K2M64Lp2L8/EFBAttnNet/model_B30.pth
8 | 30 | EFBRefineNet | 11.32 | K2M64Lp2L8/EFBRefineNet/model_B30.pth
8 | 40 | EFBAttnNet | 11.04 | K2M64Lp2L8/EFBAttnNet/model_B40.pth
8 | 40 | EFBRefineNet | 11.43 | K2M64Lp2L8/EFBRefineNet/model_B40.pth
8 | 50 | EFBAttnNet | 11.06 | K2M64Lp2L8/EFBAttnNet/model_B50.pth
8 | 50 | EFBRefineNet | 11.49 | K2M64Lp2L8/EFBRefineNet/model_B50.pth
8 | 60 | EFBAttnNet | 11.06 | K2M64Lp2L8/EFBAttnNet/model_B60.pth
8 | 60 | EFBRefineNet | 11.47 | K2M64Lp2L8/EFBRefineNet/model_B60.pth


As aforementioned, we provide model checkpoints for all the deep learning-based results. Our code library supports easy inference.

**To reproduce all these results, you need to download the given dataset and corresponding checkpoints. Also, you should arrange your projects tree as instructed.** An example of `evaluate.sh` is shown as follows.
Change the tested model using `--model` with `SubArrayEFBAttnNet` or `SubArrayRefineNet`. Change the pilot length using `-L` and change the feedback bits using `-B`.

``` bash
home=/path/to/your/home/folder
python3 ${home}/EFBAttnNet/main.py \
    --gpu 0 \
    --test-data-dir ${home}/TestData_K2M64Lp2_10000.mat \
    --model SubArrayEFBAttnNet \
    --evaluate \
    --pretrained ${home}/K2M64Lp2L4/EFBAttnNet/model_B3.pth \
    -B 3 \
    -M 64 \
    -P 1 \
    -K 2 \
    -L 4 \
    -Lp 2 \
    2>&1 | tee log.txt
```

## Acknowledgment
This repository is constructed referring to [DL-DSC-FDD](https://github.com/foadsohrabi/DL-DSC-FDD-Massive-MIMO). Thank Foad Sohrabi, Kareem M. Attiah, and Wei Yu for their excellent work. More details could be found from this [paper](https://ieeexplore.ieee.org/document/9347820).
