#!/bin/bash

# download files
echo "downloading dataset tar files"
for PART in 0 1 2 3 4 5 6 7 8 9
do
   echo "download part" $PART
   curl  https://zenodo.org/record/5528345/files/PASS.${PART}.tar --output PASS.${PART}.tar
done

# extract dataset
## will create dataset with images in PASS_dataset/dummy_folder/img-hash.jpeg
for file in *.tar; do tar -xf "$file"; done

# you can use this now e.g. with torchvision.datasets.ImageFolder('/dir/to/PASS')
