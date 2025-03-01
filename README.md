# [Semantic Segmentation for Simultaneous Crop and Land Cover Land Use Classification Using Multi-Temporal Landsat Imagery]

- [link to the Paper](https://www.sciencedirect.com/science/article/pii/S2352938525000588)
### Abstract
Land cover and land use (LCLU) mapping is essential for analyzing human-environment interactions, especially in transboundary regions where administrative borders intersect with shared resources and cultural similarities. While semantic segmentation has advanced LCLU delineation by providing robust tools for detailed mapping, further investigation is required to evaluate the potential of multispectral and multitemporal input imagery in enhancing mapping precision, particularly for agricultural field delineation. This paper employs cutting-edge semantic segmentation architectures to classify LCLU in the Middle Rio Grande (MRG) watershed, with a specific emphasis on principal crops cultivated in the area, including Alfalfa, Hay, Cotton, and Pecan. Seven models--U-Net, Feature Pyramid Network (FPN), LinkNet, DeepLabV3+, High-Resolution Network (HRNet), SegFormer, and Multi-Attention Network (MANet)—were tested. The models were adapted to incorporate multispectral Landsat 8 imagery, extending their original design intended for three-band RGB inputs. Three distinct configurations: (1) yearly median composites with Normalized Difference Vegetation Index (NDVI)- 8 bands, (2) seasonal median composites with NDVIs-32 bands (4 seasons with 8 bands each), and (3) dual-monthly median composites (July and December) with NDVIs-16 bands were tested with all models. These data configurations were designed to highlight phenological cycles and investigate their potential benefits in producing a robust LCLU map. Twenty-one models (3 datasets × 7 architecture) were trained and evaluated. Model behaviors were remarkably different for different crops and land use classes. U-Net model achieved the best performance among the group tested with RGB input (mean of per class Intersection over Union (mIoU) = 76.85%) and yearly median composite (mIoU = 78.54%) configurations. MANet exhibited had the best overall performance with dual-month (mIoU = 79.34%) and seasonal median (mIoU = 79.49%) configurations. Overall, MANet with seasonal median was the best-performing model. However, the dual-monthly was also very good and used significantly less data. The augmentation of spectral and temporal information generally enhanced model learning rates and mIoU values; however, this improvement was not uniformly observed across all architectures. This study provides empirical evidence for the feasibility and effectiveness of employing advanced semantic segmentation architectures for global large-scale, pixel-level LCLU classification. Such classifications can establish a robust foundation for informed agricultural, water resource management, and environmental decision-making while contributing to the broader understanding of land cover dynamics in complex transboundary regions. The models and datasets used in this study are available for the community to improve and apply at other sites.

## Middle Rio Grande Land Cover Land Use (LCLU) Map (2023)

View the interactive map [here](https://asu.maps.arcgis.com/home/webmap/viewer.html?webmap=7a58581ed3a142dab961bd18f0d0aa11).

![Interactive Map](https://github.com/waterdmd/Semantic_segmentation_for_LCLUC/blob/main/images/MAP.jpg)

## About This Directory

This directory encompasses the primary architectures employed in this study, including:
1. **U-Net**
2. **Feature Pyramid Network (FPN)**
3. **LinkNet**
4. **DeepLabV3+**
5. **HRNet**
6. **SegFormer**
7. **Multi-Attention Network (MANet)**

It also provides  scripts for data pre-processing and post-processing to facilitate efficient experimentation and result analysis.

Each model-specific directory contains three scripts:

1. **modeling_*.py** — Implements the training procedure for the respective model.
2. **model_inference_*.py** — Provides validation/testing routines.
3. **seg_dataset.py** — Defines the different dataset classes for various input compositions (e.g., multispectral, multitemporal data).

These scripts collectively enable end-to-end experimentation, from model training to evaluation.

**Preprocess**  
Contains modules designed for reclassifying the CDL dataset into the target classes, generating training tiles, merging multiple rasters, and retrieving data via Google Earth Engine (GEE).

**Prediction**  
Houses modules that facilitate model inference on new datasets and the creation of final output maps using the previously trained models.


## Acknowledgments
This reasearch was supported  by the [NASA Land-Cover and Land-Use Change (LCLUC) Program](https://lcluc.umd.edu/).\
This work builds upon and extends existing research and codebases from various open-source projects. We gratefully acknowledge the following repositories:

- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

We thank the respective authors and maintainers for their contributions to the open-source community.
