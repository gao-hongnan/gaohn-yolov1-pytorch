#!/bin/bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here

# Download/unzip images and labels
datasets_path='./datasets' # unzip directory
dataset_name='coco_128'
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
zip_file='coco128.zip' # or 'coco128-segments.zip', 68 MB
echo 'Downloading' $url$zip_file ' ...'
mkdir -p $d
wget -P $datasets_path/$dataset_name $url
unzip $datasets_path/$dataset_name/$zip_file -d $datasets_path/$dataset_name
rm $datasets_path/$dataset_name/$zip_file

wait # finish background tasks
