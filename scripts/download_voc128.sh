#!/bin/bash
# Download VOC128 dataset (first 128 images from VOC2007 trainval)
# Example usage: bash data/scripts/get_voc128.sh
# parent
# ├── src
# └── datasets
#     └── voc128  ← downloads here

# Download/unzip images and labels
datasets_path='./datasets' # unzip directory
dataset_name='pascal_voc_128'
url=https://storage.googleapis.com/reighns/datasets/pascal_voc_128.zip
zip_file='pascal_voc_128.zip' # zip file name
echo 'Downloading' $url$zip_file ' ...'
mkdir -p $d
wget -P $datasets_path/$dataset_name $url
unzip $datasets_path/$dataset_name/$zip_file -d $datasets_path/$dataset_name
rm $datasets_path/$dataset_name/$zip_file

wait # finish background tasks

