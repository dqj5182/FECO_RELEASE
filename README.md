<div align="center">

# FECO: Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation

<b>[Daniel Sungho Jung](https://dqj5182.github.io/)</b>, <b>[Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)</b> 

<p align="center">
    <img src="asset/logo_cvlab.png" height=55>
</p>

<b>Seoul National University</b>

<a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
<a href='https://feco-release.github.io/'><img src='https://img.shields.io/badge/Project_Page-FECO-green' alt='Project Page'></a>
<a href="https://arxiv.org/pdf/2511.22184"><img src='https://img.shields.io/badge/Paper-FECO-blue' alt='Paper PDF'></a>
<a href="https://arxiv.org/abs/2511.22184"><img src='https://img.shields.io/badge/arXiv-FECO-red' alt='Paper PDF'></a>


<h2>ArXiv 2025</h2>

<img src="./asset/teaser.png" alt="Logo" width="60%">

</div>

_**FECO** is a framework for **dense foot contact estimation** that addresses the challenges posed by **diverse shoe appearances** and **limited ground appearance variability** in foot images. Leveraging 10 datasets, including **our proposed in-the-wild dataset COFE**, we build a powerful model that learns dense foot contact across diverse scenarios._


## Code


### We are in the process of organizing the codebase, with the demo and inference code scheduled for release by the end of December and the training code in January.


## Acknowledgement
We thank:
* [SagNets](https://openaccess.thecvf.com/content/CVPR2021/papers/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.pdf) for inspiration on Shoe Style-Content Randomization.
* [Pro-RandConv](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Progressive_Random_Convolutions_for_Single_Domain_Generalization_CVPR_2023_paper.pdf) for inspiration on Low-Level Style Randomization.
* [DECO](https://openaccess.thecvf.com/content/ICCV2023/papers/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.pdf) for human-scene contact estimation.
* [HACO](https://openreview.net/pdf/675470edc47b66903daee2c8f60bd0f3fa8065f0.pdf) for dense contact estimation.



## Reference
```  
@article{jung2025feco,
    title={Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation},
    author={Jung, Daniel Sungho and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2511.22184},
    year={2025}
}
```