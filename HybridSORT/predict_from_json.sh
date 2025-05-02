#!/bin/bash


cd "$(dirname "$0")/.." | exit
# sh predict_from_json.sh --path /root/Document/data/MVA2025/pub_test
# sh predict_from_json.sh --path /root/Document/data/SMOT4SB/val
python3 tools/predict_from_json.py "$@" --save_result --min-box-area 0 --track_thres 0.1 --iou_thresh 0.1 --use_byte True --match_thresh 0.3 --asso ct_dist --aspect_ratio_thresh 4 --TCM_first_step True --TCM_byte_step True
