#!/bin/bash
for idol in 2 3 4
do
    for imgH in 640
    do
        python swapVideo.py -i data/images/rotation.MOV -o data/output/fsbench/rotation_kcf_${imgH}_high_${idol}.avi --idol ${idol} --gpu 1 --rotate 1 --highQual 1 --imgH ${imgH}
        python swapVideo.py -i data/images/expression.MOV -o data/output/fsbench/expression_kcf_${imgH}_high_${idol}.avi --idol ${idol} --gpu 1 --rotate 1 --highQual 1 --imgH ${imgH}
        python swapVideo.py -i data/images/segmentation.MOV -o data/output/fsbench/segmentation_kcf_${imgH}_high_${idol}.avi --idol ${idol} --gpu 1 --rotate 1 --highQual 1 --imgH ${imgH}
    done
done
