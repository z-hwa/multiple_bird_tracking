import os
import os.path as osp
from pathlib import Path

import torch
from loguru import logger

from trackers.ocsort_tracker.ocsort import OCSort

'''mmdet'''
from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np

'''
python OC_SORT/tools/predict_mmdet.py
'''

# 你的 Cascade R-CNN 訓練好的模型權重
model_root = "/root/Document/mva2023/"

config_file = model_root + "work_dirs/cascade_rcnn_swin_rfla_4stage_mot/cascade_rcnn_swin_rfla_4stage_mot.py"
checkpoint_file = model_root + "work_dirs/cascade_rcnn_swin_rfla_4stage_mot/epoch_20.pth"
output_dir = "Cascade_outputs/smot4sb/Swin_transformer_finetune_on_mot"  # 你可以自訂輸出資料夾
test_size = (2160, 3840)  # 讓後續處理不進行 Resize

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

def load_cascade_rcnn(config_file, checkpoint_file, device="cuda"):
    '''原始 YOLOX 代碼使用 exp.get_model() 來獲取 YOLOX 模型，你需要改為從 MMDetection 載入 Cascade R-CNN：'''

    model = init_detector(config_file, checkpoint_file, device=device)
    return model

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

def predict_videos(model, res_folder, args):
    if osp.isdir(args.path):
        video_image_dict = get_video_image_dict(args.path)
    else:
        raise ValueError(f"args.path must be a directory, but got {args.path}")

    for video_name, files in video_image_dict.items():
        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
        results = []
        for frame_id, img_path in enumerate(files, 1):
            img = cv2.imread(img_path)

            # 使用 MMDetection 進行推理
            detections = inference_detector(model, img)

            # 轉換 MMDetection 的輸出格式
            output_boxes = []
            for i, det in enumerate(detections):
                if isinstance(det, torch.Tensor):
                    det = det.cpu().numpy()  # 確保是 NumPy 陣列

                for bbox in det:
                    if bbox[4] > args.track_thresh:  # 根據置信度閾值篩選
                        x1, y1, x2, y2, score = bbox
                        output_boxes.append([x1, y1, x2, y2, score])

            # print('\n\n\n')
            # print(type(output_boxes))
            # print(output_boxes)
            # print(args.track_thresh, args.iou_thresh, args.use_byte)
            # print(img.shape[:2], test_size)
            # print('\n\n\n')
            # input()

            if output_boxes:
                online_targets = tracker.update(np.array(output_boxes), img.shape[:2], test_size)

                # print(f"Frame {frame_id}: {len(output_boxes)} detections, {len(online_targets)} tracked objects")
                # print(online_targets)
                # input()


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

    # 加載模型
    model = load_cascade_rcnn(config_file, checkpoint_file, str(args.device))

    if args.fp16:
        model = model.half()  # to FP16

    print("load successful")

    predict_videos(model, res_folder, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    
    main(args)
