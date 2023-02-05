## Overview
This is the PyTorch implementation of paper [Towards Efficient Subarray Hybrid Beamforming: Attention Network-based Practical Feedback in FDD Massive MU-MIMO System]() which has been submitted to IEEE for possible publication. The test script and trained models are listed here and the key results can be reproduced as a validation of our work.

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.8
- [PyTorch >= 1.10.1](https://pytorch.org/get-started/locally/)
- [einops](https://github.com/arogozhnikov/einops)
- [fvcore](https://github.com/facebookresearch/fvcore)


## Project Preparation

#### A. Data Preparation
The channel state information (CSI) matrix is generated according to the influential clustered Saleh Valenzuela (SV) model. The test dataset is provided [Baidu Netdisk](https://pan.baidu.com/s/1ow3F9FHSy0QeyAp94FJFEw)(passwd: gmmk), which is easy for you to download and reproduce the experiment results.
You can also generate your own dataset according to the SV channel model. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading
The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Baidu Netdisk]().

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

The main results of deep learning methods reported in our paper are presented in the following tables. All the listed results are marked in Fig. 4 and Fig. 5 in our paper. Our proposed EFBAttnNet utilizes advanced attention mechanism to achieve good performance with an extremely lightweight encoder. We also choose the end-to-end model in this [paper](https://ieeexplore.ieee.org/document/9814463) as a DL-based benchmark. Its model, which we called EFBRefineNet, is transferred into our subarray hybrid precoding scheme for fare comparison. We also provide its checkpoints in this repository.

The performance and corresponding checkpoints of Fig.4 in the paper is given as follows.

Pilot Length L | Feedback Bits B | Model | Sum Rate (Bits/s/Hz) | Checkpoint
:--: | :--: | :--: | :--: | :--:
8 | 3 | EFBAttnNet | 4.58 | K2M64Lp2L8/EFBAttnNet/model_B3.pth
8 | 3 | EFBRefineNet | 5.33 | K2M64Lp2L8/EFBRefineNet/model_B3.pth
8 | 5 | EFBAttnNet | 5.45 | K2M64Lp2L8/EFBAttnNet/model_B5.pth
8 | 5 | EFBRefineNet | 6.13 | K2M64Lp2L8/EFBRefineNet/model_B5.pth
8 | 8 | EFBAttnNet | 7.32 | K2M64Lp2L8/EFBAttnNet/model_B8.pth
8 | 8 | EFBRefineNet | 7.80 | K2M64Lp2L8/EFBRefineNet/model_B8.pth
8 | 10 | EFBAttnNet | 8.67 | K2M64Lp2L8/EFBAttnNet/model_B10.pth
8 | 10 | EFBRefineNet | 9.33 | K2M64Lp2L8/EFBRefineNet/model_B10.pth
8 | 20 | EFBAttnNet | 10.76 | K2M64Lp2L8/EFBAttnNet/model_B20.pth
8 | 20 | EFBRefineNet | 11.22 | K2M64Lp2L8/EFBRefineNet/model_B20.pth
8 | 30 | EFBAttnNet | 10.91 | K2M64Lp2L8/EFBAttnNet/model_B30.pth
8 | 30 | EFBRefineNet | 11.32 | K2M64Lp2L8/EFBRefineNet/model_B30.pth
8 | 40 | EFBAttnNet | 11.03 | K2M64Lp2L8/EFBAttnNet/model_B40.pth
8 | 40 | EFBRefineNet | 11.44 | K2M64Lp2L8/EFBRefineNet/model_B40.pth
8 | 50 | EFBAttnNet | 11.05 | K2M64Lp2L8/EFBAttnNet/model_B50.pth
8 | 50 | EFBRefineNet | 11.49 | K2M64Lp2L8/EFBRefineNet/model_B50.pth
8 | 60 | EFBAttnNet | 11.04 | K2M64Lp2L8/EFBAttnNet/model_B60.pth
8 | 60 | EFBRefineNet | 11.46 | K2M64Lp2L8/EFBRefineNet/model_B60.pth

The performance and corresponding checkpoints of Fig.5 in the paper is given as follows.

Pilot Length L| Feedback Bits B | Model | Sum Rate (Bits/s/Hz) | Checkpoint
:--: | :--: | :--: | :--: | :--:
4 | 3 | EFBAttnNet | 4.47 | K2M64Lp2L4/EFBAttnNet/model_B3.pth
4 | 3 | EFBRefineNet | 4.59 | K2M64Lp2L4/EFBRefineNet/model_B3.pth
4 | 5 | EFBAttnNet | 4.83 | K2M64Lp2L4/EFBAttnNet/model_B5.pth
4 | 5 | EFBRefineNet | 5.18 | K2M64Lp2L4/EFBRefineNet/model_B5.pth
4 | 8 | EFBAttnNet | 6.57 | K2M64Lp2L4/EFBAttnNet/model_B8.pth
4 | 8 | EFBRefineNet | 7.53 | K2M64Lp2L4/EFBRefineNet/model_B8.pth
4 | 10 | EFBAttnNet | 8.10 | K2M64Lp2L4/EFBAttnNet/model_B10.pth
4 | 10 | EFBRefineNet | 9.26 | K2M64Lp2L4/EFBRefineNet/model_B10.pth
4 | 20 | EFBAttnNet | 9.33 | K2M64Lp2L4/EFBAttnNet/model_B20.pth
4 | 20 | EFBRefineNet | 9.63 | K2M64Lp2L4/EFBRefineNet/model_B20.pth
4 | 30 | EFBAttnNet | 9.49 | K2M64Lp2L4/EFBAttnNet/model_B30.pth
4 | 30 | EFBRefineNet | 9.70 | K2M64Lp2L4/EFBRefineNet/model_B30.pth
4 | 40 | EFBAttnNet | 9.60 | K2M64Lp2L4/EFBAttnNet/model_B40.pth
4 | 40 | EFBRefineNet | 9.74 | K2M64Lp2L4/EFBRefineNet/model_B40.pth
4 | 50 | EFBAttnNet | 9.50 | K2M64Lp2L4/EFBAttnNet/model_B50.pth
4 | 50 | EFBRefineNet | 9.75 | K2M64Lp2L4/EFBRefineNet/model_B50.pth
4 | 60 | EFBAttnNet | 9.60 | K2M64Lp2L4/EFBAttnNet/model_B60.pth
4 | 60 | EFBRefineNet | 9.68 | K2M64Lp2L4/EFBRefineNet/model_B60.pth


As aforementioned, we provide model checkpoints for all the deep learning-based results. Our code library supports easy inference. *It is worth mentioning that the inference results have a certain degree of randomness brought by the random Gaussian noise in the SubarrayPilotNet.*

**To reproduce all these results, you need to download the given dataset and corresponding checkpoints. Also, you should arrange your projects tree as instructed.** An example of `evaluate.sh` is shown as follows.
Change the tested model using `--model` with `SubArrayEFBAttnNet` or `SubArrayRefineNet`. Change the pilot length using `-L` and change the feedback bits using `-B`.

``` bash
home=/path/to/your/home/folder
python3 ${home}/EFBAttnNet/main.py \
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
