# PASS: Pictures without humAns for Self-Supervised Pretraining 
**TL;DR:** An ImageNet replacement dataset for self-supervised pretraining without humans 

![img.png](img.png?style=centerme)



## Content
PASS is a large-scale image dataset that does not include any humans, human parts, or other personally identifiable information that can be used for high-quality pretraining while significantly reducing privacy concerns.

![pass.gif](pass.gif)

## Download the dataset
Generally: all information is on our [webpage](https://www.robots.ox.ac.uk/~vgg/research/pass/).

For downloading the dataset, please visit our [dataset on zenodo](https://zenodo.org/record/5501843). There you can download it in tar files and find the meta-data.

You can also download the images from their AWS urls, from [here](https://www.robots.ox.ac.uk/~vgg/research/pass/pass_urls.txt).

## Pretrained models
| Pretraining | Method                                                              | Epochs | Places205 lin. Acc. | Model weights                                                                                                              |
|-------------|------------------------------------------------------------------------|--------|---------------------|------------------------------------------------------------------------------------------------------------------|
| <span style="color:grey">IN-1k</span>      | [<span style="color:grey">MoCo-v2</span> ](https://github.com/facebookresearch/moco)                   | <span style="color:grey">200</span>    | <span style="color:grey">50.1                |  [<span style="color:grey">R50 weights</span>](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar)|
| PASS        | [MoCo-v2](https://github.com/facebookresearch/moco)                    | 200    | 52.8                | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_200ep.pth.tar)            |
| PASS        | [MoCo-v2-CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning) | 200    | 53.1                | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/moco_v2_CLD_200ep.pth.tar)        |
| PASS        | [SwAV](https://github.com/facebookresearch/swav)                       | 200    | 55.5                | [R50 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/swav_200ep.pth.tar)               |
| PASS        | [DINO](https://github.com/facebookresearch/dino)                       | 100    | X                   | [ViT S16 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_100ep.pth.tar)      |
| PASS        | [DINO](https://github.com/facebookresearch/dino)                       | 300    | X                    | [ViT S16 weights](https://www.robots.ox.ac.uk/~vgg/research/pass/pretrained_models/dino_deit_300ep_ttemp0o07_warumup30ep_normlayerF.pth.tar)                                                                                                  |
| PASS        | [MoCo-v2](https://github.com/facebookresearch/moco)                    | 800    |                     | coming soon                                                                                                      |                                                                                             |                               |

## Pretrained models from PyTorch Hub
```
import torch
vits16_100ep = torch.hub.load('yukimasano/PASS:main', 'dino_100ep_vits16')
vits16 = torch.hub.load('yukimasano/PASS:main', 'dino_vits16')
r50_swav = torch.hub.load('yukimasano/PASS:main', 'swav_resnet50')
r50_moco = torch.hub.load('yukimasano/PASS:main', 'moco_resnet50')
r50_moco_cld = torch.hub.load('yukimasano/PASS:main', 'moco_cld_resnet50')
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
