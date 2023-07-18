# Adaptive Multimodal Prompt for Human-Object Interaction with Local Feature Enhanced Transformer (AMP-HOI)

## Overview

AMP-HOI is an end-to-end transformer-based and cnn-based human-object interaction (HOI) detector. [[Paper]]()

![AMP-HOI](./figures/AMP-HOI_arch.png)

- **Motivation**: (1) The loss of crucial features from the original modality during contrastive learning. (2) The limited ability of Transformer-based network architectures to extract local features from samples. (3) There is still room for improvement in the application of prompt learning on HOI.
- **Components**: (1) We proposed an Adaptive Multimodal Prompt module that facilitates the interaction of multimodal cues and provides specific and applicable cues for different modalities. (2) We introduced a novel multimodal feature extraction module called the Local Feature Enhanced Transformer (LFET), which effectively extracts multimodal features from both global and local perspectives.

## Preparation

### Installation

Our code is built upon [CLIP](https://github.com/openai/CLIP). This repo requires to install [PyTorch](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies.

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install ftfy regex tqdm numpy Pillow matplotlib
```

### Dataset

The experiments are mainly conducted on **HICO-DET** dataset. We follow [this repo](https://github.com/YueLiao/PPDM) to prepare the HICO-DET dataset.

#### HICO-DET

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory. We use the annotation files provided by the [PPDM](https://github.com/YueLiao/PPDM) authors. We re-organize the annotation files with additional meta info, e.g., image width and height. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg). The downloaded files have to be placed as follows. Otherwise, please replace the default path to your custom locations in [datasets/hico.py](./datasets/hico.py).

``` plain
 |─ data
 │   └─ hico_20160224_det
 |       |- images
 |       |   |─ test2015
 |       |   |─ train2015
 |       |─ annotations
 |       |   |─ trainval_hico_ann.json
 |       |   |─ test_hico_ann.json
 :       :
```

## Training

Run this command to train the model in HICO-DET dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --batch_size 8 \
    --output_dir [path to save checkpoint] \
    --epochs 30 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 10 \
    --enable_dec \
    --enable_resnet50 \
    --enable_gru \
    --enable_text_lambda \
    --enable_visual_lambda1 \
    --enable_visual_lambda2 \
    --lamb 0.6 \
    --enable_unified_prompt \
    --dataset_file hico
```

## Inference

Run this command to evaluate the model on HICO-DET dataset

``` bash
python main.py --eval \
    --batch_size 1 \
    --output_dir [path to save results] \
    --hoi_token_length 10 \
    --enable_dec \
    --pretrained [path to the pretrained model] \
    --eval_size 256 [or 224 448 ...] \
    --test_score_thresh 1e-4 \
    --enable_resnet50 \
    --enable_gru \
    --enable_text_lambda \
    --enable_visual_lambda1 \
    --enable_visual_lambda2 \
    --lamb 0.6 \
    --enable_unified_prompt \
    --dataset_file hico
```

## Models

|   Model   | dataset | HOI Tokens | AP seen | AP unseen | Log | Checkpoint |
|:---------:| :-----: | :-----: |:-------:|:---------:| :-----: | :-----: |
| `AMP-HOI` | HICO-DET | 10 |  23.33  |   21.75   | [Log](https://github.com/scwangdyd/promting_hoi/releases/download/v0.2/thid_hico_token10_epoch100_log.txt) | [params](https://github.com/scwangdyd/promting_hoi/releases/download/v0.2/thid_hico_token10_epoch100.pth)|

## Citing

Please consider citing our paper if it helps your research.

```
None
```