# Phase Segmentation of SOC SEM Images using U-Net
This repository implements a deep learning pipeline for **semantic segmentation of Solid Oxide Cell (SOC)** microstructures from **2D SEM images**, without relying on EDS data at inference time.



#### Objective

Accurate phase identification of:

- **Porosity**
- **Metallic Nickel (Ni)**
- **Nickel Oxide (NiO)**
- **Yttria-Stabilized Zirconia (YSZ)**

using **only SEM images**, bypassing time-consuming and equipment-intensive **Energy Dispersive X-ray Spectroscopy (EDS)**.

![EDS](\\kiev\share_nfs\200-Science_et_Technique\200.4-PFNC-DATA\kd264511\Github\U-Net-SOFC\Figure\EDS.png)





#### Methodology



![Workflow-1](\\kiev\share_nfs\200-Science_et_Technique\200.4-PFNC-DATA\kd264511\Github\U-Net-SOFC\Figure\Workflow-1.png)

**(a). Data Acquisition and Pre-processing**
SEM and EDS images were acquired from Ni/YSZ hydrogen electrodes of SOCs using a Zeiss Gemini SEM 460 with an Oxford EDS detector. EDS maps, collected at lower resolution to reduce drift and acquisition time, were later upsampled and aligned to SEM images via a custom Python script. Ground truth segmentation was generated using Ilastik, based on semi-automated annotations. To compensate for limited labeled data, 1232 image patches (256×256) were extracted and augmented with transformations, expanding the dataset to 10,920 samples for training.

**(b). training the U-Net architecture**

A U-Net model was trained in a supervised manner to classify four key phases: porosity, Ni, NiO, and YSZ. Standard cross-entropy loss was used, with data augmentation and tuning strategies enhancing robustness. The model learned to accurately predict segmentation maps from raw SEM images—without requiring EDS at inference.

**(c). U-Net segmentation of SEM images**
Full-size images were segmented using a sliding window approach with overlapping patches. Predictions were averaged in overlapping regions to reduce edge artifacts and improve continuity, yielding high-quality, scalable segmentations suitable for automated microstructural analysis.

