# 3D Liver Tumor Segmentation (UNet + TorchIO + PyTorch Lightning)

![Result](result/result%20evaluation.gif)

3D liver + tumor segmentation from CT volumes using a 3D U-Net trained with **Cross-Entropy + Dice Loss**, patch-based sampling (TorchIO Queue), and PyTorch Lightning.

## Dataset

This project uses the **Medical Segmentation Decathlon (MSD)** dataset:

- **Task03_Liver** (CT volumes + liver/tumor labels)
- Download from: **medicaldecathlon.com**

## Method

- **Model:** 3D UNet (see `model.py`)
- **Loss:** `CrossEntropyLoss(weight=[1,1,3]) + DiceLoss`
- **Preprocessing:**
  - `CropOrPad((256, 256, 200))`
  - `RescaleIntensity((0, 1))`
- **Training strategy:**
  - Patch sampling with `tio.Queue` + `LabelSampler`
  - Augmentation: `RandomAffine(scales=(0.9, 1.1), degrees=10)`
- **Optimization:**
  - AdamW, `lr=1e-4`
  - ReduceLROnPlateau on `val_loss`
- **Logging:** TensorBoard + validation overlay figure
