# FracFormer
FracFormer is a Transformer-based network designed for fracture reduction planning, utilizing point cloud data from fracture segmentation to estimate the target pose of bone fragments. It formulates the task as a patch-to-patch translation problem to reconstruct bone shapes. The network includes a Fragment-Aware Patch Encoding (FAPE) module for extracting local features and embedding fragment labels, a patch morphing module for predicting reduction deformations, and a reconstruction module for generating dense point clouds to restore precise bone shapes. The repository provides example datasets for various bone types and training code for the FracFormer network.

![image](https://github.com/Sutuk/FracFormer/blob/main/data/overview.png)
