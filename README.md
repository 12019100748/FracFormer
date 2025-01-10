# FracFormer
FracFormer is a Transformer-based network designed for fracture reduction planning, utilizing point cloud data from fracture segmentation to estimate the target pose of bone fragments. It formulates the task as a “patch-to-patch translation” problem to reconstruct bone shapes. The network features a Fragment-Aware Patch Encoding (FAPE) module for extracting local features and embedding fragment labels, a patch morphing module for predicting geometric deformations and query embeddings, and a reconstruction module for generating dense point clouds to achieve precise bone shape restoration. The repository includes example datasets for various bone types and training code for the FracFormer network.

![image](https://github.com/Sutuk/FracFormer/blob/main/data/overview.png)
