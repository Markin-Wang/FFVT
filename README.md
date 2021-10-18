# MASK GUIDED ATTENTION FOR FINE-GRAINED PATCHY IMAGE CLASSIFICATION

PyTorch implementation of 

[MASK GUIDED ATTENTION FOR FINE-GRAINED PATCHY IMAGE CLASSIFICATION](https://arxiv.org/pdf/2102.02771.pdf)" ( ICIP 2021 ) 

If you use the code in this repo for your work, please cite the following bib entries:

    @INPROCEEDINGS{9506424,
      author={Wang, Jun and Yu, Xiaohan and Gao, Yongsheng},
      booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
      title={Mask Guided Attention For Fine-Grained Patchy Image Classification}, 
      year={2021},
      volume={},
      number={},
      pages={1044-1048},
      doi={10.1109/ICIP42928.2021.9506424}
      }


## Abstract

In this work, we present a novel mask guided attention (MGA) method for fine-grained patchy image classification. The key challenge of fine-grained patchy image classification lies in two folds, ultra-fine-grained inter-category variances among objects and very few data available for training. This motivates us to consider employing more useful supervision signal to train a discriminative model within limited training samples. Specifically, the proposed MGA integrates a pre-trained semantic segmentation model that produces auxiliary supervision signal, i.e., patchy attention mask, enabling a discriminative representation learning. The patchy attention mask drives the classifier to filter out the insignificant parts of images (e.g., common features between different categories), which enhances the robustness of MGA for the fine-grained patchy image classification. We verify the effectiveness of our method on three publicly available patchy image datasets. Experimental results demonstrate that our MGA method achieves superior performance on three datasets compared with the state-of-the-art methods. In addition, our ablation study shows that MGA improves the accuracy by 2.25% and 2% on the SoyCultivarVein and BtfPIS datasets, indicating its practicality towards solving the fine-grained patchy image classification.

<img src='architecture.png' width='1280' height='350'>


## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch >= 1.0]
- [Torchvision]

## Dataset
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1EF_iamMlnb0QYS2xiQRq--fxm7an4tv7?usp=sharing).

soybean_1_1: The soycultivarvein dataset with the training set:test set=1:1. For comparison to the state-of-the-art hand-crafted methods.

soybean_2_1: The soycultivarvein dataset with the training set:test set=2:1.


## Training scripts for MGANet with the backbone network Densenet161.
Train the model on the Soybean dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 32.

    $ python train.py --dataset soybean_2_1 --lr 0.05 --backbone_class densenet161


## Testing scripts for MGANet with the backbone network Densenet161.
Test the model on the Soybean dataset:

    $ python test.py  --dataset soybean_2_1 --backbone_class densenet161
    
        
            
## Download  Models


[Trained model Google Drive](https://drive.google.com/drive/folders/11SA7PGR9NbyJEaXFOHwA_PGiORdIEoYZ?usp=sharing)

## Segmentation Experiments.
For the leaf vein segmentation experiments, please refer to [Nvidia/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation) to gain the details.

All the three datasets are trained with the crop size (448,448) and 60 epochs.



## Acknowledgment
Thanks for the advice and guidance given by Dr.Xiaohan Yu and Prof. Yongsheng Gao.

Our project references the codes in the following repos.
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [Nvidia/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)




