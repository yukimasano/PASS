# PASS: Pictures without humAns for Self-Supervised Pretraining 
**TL;DR:** An ImageNet replacement dataset for self-supervised pretraining without humans 

![img.png](img.png?style=centerme)



## Content
PASS is a large-scale image dataset that does not include any humans, human parts, or other personally identifiable information that can be used for high-quality pretraining while significantly reducing privacy concerns.

![pass.gif](pass.gif)

## Download the dataset

The quickest way:
```sh
git clone https://github.com/yukimasano/PASS
cd PASS
source download.sh # maybe change the directory where you want to download it
```
Generally: all information is on our [webpage](https://www.robots.ox.ac.uk/~vgg/research/pass/).

For downloading the dataset, please visit our [dataset on zenodo](https://zenodo.org/record/5528345). There you can download it in tar files and find the meta-data.

You can also download the images from their AWS urls, from [here](https://www.robots.ox.ac.uk/~vgg/research/pass/pass_urls.txt).

## Pretrained models
| Pretraining | Method                                                                 | Epochs | IN-1k Acc. | Places205 Acc. |                                                                                                                                              |
|-------------|------------------------------------------------------------------------|--------|------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| (IN-1k)     | [MoCo-v2 ](https://github.com/facebookresearch/moco)                   | 200    | 60.6       | 50.1           | [visit MoCo-v2 repo](https://github.com/facebookresearch/moco#models)                                                                        |
| PASS        | [MoCo-v2](https://github.com/facebookresearch/moco)                    | 180    | 59.1       | 52.8           | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_180ep_of200ep.pth.tar)                                |
| PASS        | [MoCo-v2](https://github.com/facebookresearch/moco)                    | 200    | 59.5       | 52.8           | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_200ep.pth.tar)                                        |
| PASS        | [MoCo-v2](https://github.com/facebookresearch/moco)                    | 800    | 61.2       | XX             | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_800ep.pth.tar)                                        |
| PASS        | [MoCo-v2-CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning) | 200    | 60.2       | 53.1           | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_CLD_200ep.pth.tar)                                    |
| PASS        | [SwAV](https://github.com/facebookresearch/swav)                       | 200    | 60.8       | 55.5           | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/swav_200ep.pth.tar)                                           |
| PASS        | [DINO](https://github.com/facebookresearch/dino)                       | 100    | 61.3       | 54.6           | [ViT S16 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_100ep.pth.tar)                                  |
| PASS        | [DINO](https://github.com/facebookresearch/dino)                       | 300    | 65.0       | 55.7           | [ViT S16 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_300ep_ttemp0o07_warumup30ep_normlayerF.pth.tar) |
|             |                                                                        |        |            |                |                                                                                                                                              |

In the table above we give the download links to the full checkpoints (including momentum encoder etc.) to the models we've trained. 
For comparison, we include MoCo-v2 trained on ILSVRC-12 ("IN-1k") and report linear probing performance on IN-1k and Places205.

## Pretrained models from PyTorch Hub
```python
import torch
vits16_100ep = torch.hub.load('yukimasano/PASS:main', 'dino_100ep_vits16')
vits16 = torch.hub.load('yukimasano/PASS:main', 'dino_vits16')
r50_swav_200ep = torch.hub.load('yukimasano/PASS:main', 'swav_resnet50')
r50_moco_800ep = torch.hub.load('yukimasano/PASS:main', 'moco_resnet50')
r50_moco_cld_200ep = torch.hub.load('yukimasano/PASS:main', 'moco_cld_resnet50')
```  
  
### Contribute your models

Please let us know if you have a model pretrained on this dataset and I will add this to the list above.

## Citation
```
@Article{asano21pass,
author = "Yuki M. Asano and Christian Rupprecht and Andrew Zisserman and Andrea Vedaldi",
title = "PASS: An ImageNet replacement for self-supervised pretraining without humans",
journal = "NeurIPS Track on Datasets and Benchmarks",
year = "2021"
} 
```
