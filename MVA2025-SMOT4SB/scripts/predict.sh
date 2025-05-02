#!/bin/bash


cd "$(dirname "$0")/.." | exit

# sh scripts/predict.sh -f OC_SORT/exps/smot4sb.py --path OC_SORT/datasets/SMOT4SB/pub_test --ckpt YOLOX_outputs/smot4sb/best_ckpt.pth.tar
python3 OC_SORT/tools/predict.py "$@" --save_result --min-box-area 0 --track_thres 0.1 --iou_thresh 0.1 --use_byte True --match_thresh 0.3 --asso diou --track_buffer 50 --min_hits 1 --deltat 7 --aspect_ratio_thresh 4
