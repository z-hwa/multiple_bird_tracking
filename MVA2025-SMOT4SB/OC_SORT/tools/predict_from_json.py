import os
import os.path as osp
from pathlib import Path

import torch
from loguru import logger

from trackers.ocsort_tracker.ocsort import OCSort

import cv2
import numpy as np

import json


test_size = (2160, 3840)  # 讓後續處理不進行 Resize

# 載入 COCO 預測和標註 JSON 檔案
# pub test phase 2
# output_dir = "Cascade_outputs/phase_2/past_fut_feature-boost"  # 你可以自訂輸出資料夾
# predictions_json_path = "../mva2023/" + "work_dirs/stack_cascade_rcnn_v2_swin_rfla_4stage/results_f10_smot4sb_phase2.bbox.json"
# annotations_json_path = "/root/Document/data/MVA2025/annotations/test_coco.json"

# val
output_dir = "Cascade_outputs/val/filtered_results_smot4sb_val"  # 你可以自訂輸出資料夾
predictions_json_path = "../mva2023/" + "filtered_results_smot4sb_val.json"
annotations_json_path = "/root/Document/data/SMOT4SB/annotations/val.json"

# other setting
converted_results = None
c_id = 1
asso_func = 'diou'

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

def load_json(file_path):
    """載入 JSON 檔案"""
    with open(file_path, "r") as f:
        return json.load(f)

def convert_coco_predictions(coco_predictions, empty_annotations):
    """
    將 COCO 預測結果轉換為 {file_name: tensor_list} 結構。
    
    :param coco_predictions: COCO 格式的預測結果 (list of dicts)
    :param empty_annotations: 原始空標註資料，應包含 {"images": [{"id": X, "file_name": "path/to/image"}, ...]}
    :return: dict，key 為 file_name，value 為該圖片的預測結果 tensor list
    """
    # 建立 image_id 到 file_name 的映射
    image_id_to_file = {img["id"]: img["file_name"] for img in empty_annotations["images"]}
    
    # 建立 {file_name: list of tensors} 結構
    results = {}
    for pred in coco_predictions:
        image_id = pred["image_id"]
        file_name = image_id_to_file.get(image_id, None)
        if file_name is None:
            continue
        
        bbox = pred["bbox"]  # [x, y, w, h]
        score = pred["score"]
        category_id = pred["category_id"]

        # 轉換為 [x1, y1, x2, y2, score]，符合 MMDetection 格式
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        bbox_data = np.array([x1, y1, x2, y2, score])

        # 初始化該圖片的分類列表
        if file_name not in results:
            results[file_name] = {}

        # 把 bbox 存入對應類別的 list
        if category_id not in results[file_name]:
            results[file_name][category_id] = []
        
        results[file_name][category_id].append(bbox_data)
    
    # 把 list 轉成 numpy array，確保格式一致
    for file_name in results:
        for category_id in results[file_name]:
            results[file_name][category_id] = np.array(results[file_name][category_id])
    
    return results

def get_video_image_dict(root_path):
    video_image_dict = {}
    
    for video_name in os.listdir(root_path):
        video_path = osp.join(root_path, video_name)
        
        if not osp.isdir(video_path):
            continue

        image_paths = []
        for maindir, _, file_name_list in os.walk(video_path):
            for filename in file_name_list:
                ext = osp.splitext(filename)[1].lower()
                if ext in IMAGE_EXT:
                    image_paths.append(osp.join(maindir, filename))

        video_image_dict[video_name] = sorted(image_paths)

    return video_image_dict

def predict_videos(res_folder, args):
    if osp.isdir(args.path):
        video_image_dict = get_video_image_dict(args.path)
    else:
        raise ValueError(f"args.path must be a directory, but got {args.path}")

    for video_name, files in video_image_dict.items():
        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=0.1, use_byte=args.use_byte,
                        asso_func=asso_func, inertia=0.4)
        results = []
        for frame_id, img_path in enumerate(files, 1):
            img = cv2.imread(img_path)

            # 查詢預測結果表 進行推理
            sub_path = "/".join(img_path.split('/')[-2:])
            
            # 檢查 key 是否存在，若無則跳過該影像
            if sub_path not in converted_results:
                continue

            detections = [converted_results[sub_path][c_id]] #inference_detector(model, img)

            # 轉換 MMDetection 的輸出格式
            output_boxes = []

            for i, det in enumerate(detections):
                if isinstance(det, torch.Tensor):
                    det = det.cpu().numpy()  # 確保是 NumPy 陣列

                for bbox in det:
                    if bbox[4] > args.track_thresh:  # 根據置信度閾值篩選
                        x1, y1, x2, y2, score = bbox
                        output_boxes.append([x1, y1, x2, y2, score])

            if output_boxes:
                online_targets = tracker.update(np.array(output_boxes), img.shape[:2], test_size)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n"
                        )

            if frame_id % 20 == 0:
                logger.info('Processing frame {}'.format(frame_id))

        if args.save_result:
            res_file = osp.join(res_folder, f"{video_name}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

def main(args):
    
    if not args.path:
        raise ValueError("Please specify the path to the subset directory")
    
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        res_folder = osp.join(output_dir, "predictions", Path(args.path).stem)
        os.makedirs(res_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    logger.info("Args: {}".format(args))

    print("load successful")

    predict_videos(res_folder, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    
    coco_predictions = load_json(predictions_json_path)
    annotations = load_json(annotations_json_path)

    # 轉換結果
    converted_results = convert_coco_predictions(coco_predictions, annotations)

    main(args)


