#!/bin/bash


cd "$(dirname "$0")/.." | exit
# sh scripts/predict_from_json.sh --path /root/Document/data/MVA2025/pub_test
# sh scripts/predict_from_json.sh --path /root/Document/data/SMOT4SB/val
python3 OC_SORT/tools/predict_from_json.py "$@" --save_result --min-box-area 0 --track_thres 0.1 --iou_thresh 0.1 --use_byte True --match_thresh 0.3 --asso ct_dist --track_buffer 50 --min_hits 1 --deltat 7 --aspect_ratio_thresh 4
