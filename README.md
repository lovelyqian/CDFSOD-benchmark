# 1 Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector


- [**News!**] 24-07-01: Our work is accepted by ECCV24. [Arxiv Paper](https://arxiv.org/pdf/2402.03094) can be found here. 

- [**News!**] 24-07-12: We build our [Project Page](http://yuqianfu.com/CDFSOD-benchmark) which includes a brief summary of our work.

- [**News!**] 24-07-13: We released the datasets and also codes. Welcome to use this benchmark and also try our proposed method!

<img width="470" alt="image" src="http://yuqianfu.com/images/cdfsod-benchmark.jpg">

**In this paper**, we: 
1) reorganize a **benchmark** for Cross-Domain Few-Shot Object Detection (CD-FSOD);
2) conduct **extensive study** on several different kinds of detectors (Tab.1 in the paper);
3) propose a novel **CD-ViTO** method via enhancing the existing open-set detector (DE-ViT).

**In this repo**, we provide: 
1) links and splits for target datasets;
2) codes for our CD-ViTO method;
3) codes for the DE-ViT-FT method; (in case you would like to build new methods based on this baseline).


# 2 Datasets
In this benchmark, we take COCO as source training data and use ArTaxOr, Clipart1k, DIOR, DeepFish, NEU-DET, and UODD as target datasets. 
![image](https://github.com/user-attachments/assets/a56cb01e-fb06-4528-b63d-a373240572da)

Also, in this paper, we adopt the "pretrain, finetuning, and testing" pipeline, while the pre-trained stage on COCO is directly taken from the [DE-ViT], thus in practice, only the targets are needed to run our experiments.  

The target datasets could be easily downloaded in the following links:  (If you use the datasets, please cite the corresponding papers properly, thanks.)

- [Dataset Link from 百度云盘](https://pan.baidu.com/s/1MpTwmJQF6GtmnxauVUPNAw?pwd=ni5j)
- [Dataset Link from Google Drive （coming soon）]()


# 3 Methods
## 3.1 Setup
An anaconda environment is suggested, take the name "cdfsod" as an example: 

```
git clone git@github.com:lovelyqian/CDFSOD-benchmark.git
conda create -n cdfsod python=3.9
conda activate cdfsod
pip install -r CDFSOD-benchmark/requirements.txt 
pip install -e ./CDFSOD-benchmark
cd CDFSOD-benchmark
```

## 3.2 Run CD-ViTO


weights:
download weights from [devit](https://github.com/mlzxy/devit/blob/main/Downloads.md)


## 3.3 Run DE-ViT-FT

# 4 Citiing
If you find our paper or this code useful for your research, please considering cite us (●°u°●)」:
```
@article{fu2024cross,
  title={Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector},
  author={Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu and Shangguan, Zeyu and Liu, Tong and Kong, Lingjie and Fu, Yanwei and Van Gool, Luc and others},
  journal={arXiv preprint arXiv:2402.03094},
  year={2024}
}
```


