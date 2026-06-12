# DCS-UNet: Cross-Scale Feature Aggregation with Adaptive Decoder Refinement for Camouflaged Object Segmentation
## Introduction
DCS-UNet is a SAM2-based segmentation framework with lightweight task-specific adaptation for camouflaged object segmentation (COS). The framework addresses two core challenges: insufficient feature discriminability in ambiguous foreground–background scenes, and semantic inconsistency introduced by direct skip-connection fusion.
## Requirements
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install timm einops pytorch_wavelets
```
## Training
```bash
python train.py \
    --hiera_path "sam2_configs/sam2_hiera_large.pt" \
    --train_image_path "data/TrainDataset/image" \
    --train_mask_path "data/TrainDataset/mask" \
    --save_path "output" \
    --epoch 50 \
    --lr 0.001 \
    --batch_size 12
```
## Testing
```bash
python test.py \
    --checkpoint "output/DCS-50.pth" \
    --test_image_path "data/TestDataset/Image" \
    --test_gt_path "data/TestDataset/GT" \
    --save_path "output_test/Image"
```
## Evaluation
```bash
python eval.py \
    --dataset_name "COD10K" \
    --pred_path "output_test/Image" \
    --gt_path "data/TestDataset/GT"
```
## Data
Please download the datasets from their official sources:
- **COD10K**: [Download](https://github.com/DengPingFan/SINet)
- **CAMO**: [Download](https://sites.google.com/view/ltnghia/research/camo)
- **CHAMELEON**: [Download](https://www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis/)
- **NC4K**: [Download](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)
- **PlantCamo**: [Download](https://github.com/yjybuaa/PlantCamo)

