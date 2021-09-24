#!/bin/bash

# download files
echo "downloading dataset tar files"
for PART in 1 2 3 4 5 6 7 8 9
do
   echo "download part" $PART
   curl  https://zenodo.org/record/5501843/files/PASS.tar.part.${PART} --output PASS.tar.part.${PART}
done

# concatenate parts
cat PASS.tar.part* > PASS.tar

# extract dataset
## will create dataset with images in PASS/dummy_folder/img-hash.jpeg
tar -xf PASS.tar

# you can use this now e.g. with torchvision.datasets.ImageFolder('/dir/to/PASS')
