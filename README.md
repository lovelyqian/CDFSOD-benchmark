# Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector
- [**News!**] 25-02-25: Based on this work, we are organzing the **[1st CD-FSOD Challenge](https://codalab.lisn.upsaclay.fr/competitions/21851 
)** under [NTIRE Workshop @ CVPR 2025](https://www.cvlai.net/ntire/2025/).  Top participants could win awards and publish papers on **CVPR25 workshop**. 🏆 

- [**News!**] 24-07-01: Our work is accepted by ECCV24. [Arxiv Paper](https://arxiv.org/pdf/2402.03094) can be found here. 🎉 

- [**News!**] 24-07-12: We build our [Project Page](http://yuqianfu.com/CDFSOD-benchmark) which includes a brief summary of our work. 🔥

- [**News!**] 24-07-13: We released the [Datasets](https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q?usp=drive_link) and also [Codes](https://github.com/lovelyqian/CDFSOD-benchmark). Welcome to use this benchmark and also try our proposed method! 🌟

- [**News!**] 24-07-16: We build the leaderboards on [Paper With Code: Cross-Domain Few-Shot Object Detection](https://paperswithcode.com/task/cross-domain-few-shot-object-detection/latest). 🥂

- [**News!**] 24-09-16: We uploaded the presentation videos at [Bilibili: English Pre](https://www.bilibili.com/video/BV17v4UetEdF/?spm_id_from=333.999.0.0), [Bilibili: 中文讲解](https://www.bilibili.com/video/BV11etbenET7/?spm_id_from=333.999.0.0&vd_source=668a0bb77d7d7b855bde68ecea1232e7), [Youtube: English Pre](https://www.youtube.com/watch?v=t5vREYQIup8). 😊


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on-1)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-1?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on-3)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-3?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on-2)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-2?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on-neu)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-neu?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/cross-domain-few-shot-object-detection-on-4)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-4?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/few-shot-object-detection-on-ms-coco-10-shot)](https://paperswithcode.com/sota/few-shot-object-detection-on-ms-coco-10-shot?p=cross-domain-few-shot-object-detection-via)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-few-shot-object-detection-via/few-shot-object-detection-on-ms-coco-30-shot)](https://paperswithcode.com/sota/few-shot-object-detection-on-ms-coco-30-shot?p=cross-domain-few-shot-object-detection-via)


**In this paper**, we: 
1) reorganize a **benchmark** for Cross-Domain Few-Shot Object Detection (CD-FSOD);
2) conduct **extensive study** on several different kinds of detectors (Tab.1 in the paper);
3) propose a novel **CD-ViTO** method via enhancing the existing open-set detector (DE-ViT).

**In this repo**, we provide: 
1) links and splits for target datasets;
2) codes for our CD-ViTO method;
3) codes for the DE-ViT-FT method; (in case you would like to build new methods based on this baseline).


# Datasets
We take **COCO** as source training data and **ArTaxOr**, **Clipart1k**, **DIOR**, **DeepFish**, **NEU-DET**, and **UODD** as targets. 

![image](https://github.com/user-attachments/assets/532dc8db-47eb-4e84-be46-7a59f8ff0461)


Also, as stated in the paper, we adopt the "pretrain, finetuning, and testing" pipeline, while the pre-trained stage on COCO is directly taken from the [DE-ViT](https://github.com/mlzxy/devit), thus in practice, only the targets are needed to run our experiments.  

The target datasets could be easily downloaded in the following links:  (If you use the datasets, please cite them properly, thanks.)

- [Dataset Link from Google Drive](https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q?usp=drive_link)
- [Dataset Link from 百度云盘](https://pan.baidu.com/s/1MpTwmJQF6GtmnxauVUPNAw?pwd=ni5j)

To train CD-ViTO on a custom dataset, please refer to [DATASETS.md](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/DATASETS.md) for detailed instructions.

# Methods
## Setup
An anaconda environment is suggested, take the name "cdfsod" as an example: 

```
git clone git@github.com:lovelyqian/CDFSOD-benchmark.git
conda create -n cdfsod python=3.9
conda activate cdfsod
pip install -r CDFSOD-benchmark/requirements.txt 
pip install -e ./CDFSOD-benchmark
cd CDFSOD-benchmark
```

## Run CD-ViTO
1. download weights:
download pretrained model from [DE-ViT](https://github.com/mlzxy/devit/blob/main/Downloads.md).

2. run script: 
```
bash main_results.sh
```


## Run DE-ViT-FT
Add --controller to main_results.sh, then
```
bash main_results.sh
```

# Acknowledgement

Our work is built upon [DE-ViT](https://github.com/mlzxy/devit), and also we use the codes of [ViTDeT](https://github.com/ViTAE-Transformer/ViTDet), [Detic](https://github.com/facebookresearch/Detic) to test them under this new benchmark. Thanks for their work.

# Citation
If you find our paper or this code useful for your research, please considering cite us (●°u°●)」:
```
@inproceedings{fu2025cross,
  title={Cross-domain few-shot object detection via enhanced open-set object detector},
  author={Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu and Shangguan, Zeyu and Liu, Tong and Fu, Yanwei and Van Gool, Luc and Jiang, Xingqun},
  booktitle={European Conference on Computer Vision},
  pages={247--264},
  year={2025},
  organization={Springer}
}
```

(we also have several works on cross-domain few-shot learning, if you are generally interested in this topic, a citation is very appreciated) 

```
@inproceedings{fu2023styleadv,
  title={Styleadv: Meta style adversarial training for cross-domain few-shot learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24575--24584},
  year={2023}
}

@inproceedings{fu2021meta,
  title={Meta-fdmixup: Cross-domain few-shot learning guided by labeled target data},
  author={Fu, Yuqian and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the 29th ACM international conference on multimedia},
  pages={5326--5334},
  year={2021}
}
```


