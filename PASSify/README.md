# PASSify your dataset
Here we provide the automated scripts that remove humans from your dataset using a face detector and a person detector.
This automated procedure does **not** guarantee full exclusion of humans or personal identifiable information (e.g. licence plates will still be present, some humans might slip through). 
However with this you can find:
* how much of your dataset roughly includes humans and human faces
* how much your model performance changes when trained on the PASSified version of your dataset.

## Running instructions
For all of the following, we provide instructions for running commands on a SLURM managed cluster, but you can tailor them to run on a single machine too.
You very likely need to adapt the slurm headers slightly to fit your cluster, you can find them in `passify.py`.
Have your dataset of images in in a structure as `/path/to/dataset/{folders}/{imagename}`.

1. Face detector
We start with the face detector as this one is cheaper to run and can be run on CPUs.
    ```sh
    DATA_DIRECTORY=/path/to/dataset/
    python passify.py 0 $DATA_DIRECTORY
    sbatch face/sbatch_face.sh
    ```

2. Person detector
    Next we run the person detector on GPUs. For this you need to have installed the [detectron2 repo](https://github.com/facebookresearch/detectron2).
    ```sh
    
    python passify.py 1
    sbatch person/sbatch_person.sh
    ```

3. Final list
   Finally count the files that you're left with.
    ```sh
    python passify.py 2
    ```

## References
This work relies on two excellent repos:

The facedetector is from [Retinaface](https://github.com/biubug6/Pytorch_Retinaface) (MIT Licence)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
The person detector is from [detectron2 repo](https://github.com/facebookresearch/detectron2) (Apache Licence), specifically, the Cascade-RCNN trained with 3x.
```
@misc{wu2019detectron2,
author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
              Wan-Yen Lo and Ross Girshick},
title =        {Detectron2},
howpublished = {\url{https://github.com/facebookresearch/detectron2}},
year =         {2019}
}
```
## Citation
If you found this useful please consider citing
```
@Article{asano21pass,
author = "Yuki M. Asano and Christian Rupprecht and Andrew Zisserman and Andrea Vedaldi",
title = "PASS: An ImageNet replacement for self-supervised pretraining without humans",
journal = "NeurIPS Track on Datasets and Benchmarks",
year = "2021"
}
```
