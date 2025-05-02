import os
import os.path as osp
from pathlib import Path

import torch
from loguru import logger

from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

'''
python OC_SORT/tools/my_tools/det_checker.py --path OC_SORT/datasets/SMOT4SB/pub_test --track_thresh 0.1
'''

config_file = "/root/Document/mva2023/configs/_MyPlan/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_nwd.py"
checkpoint_file = "/root/Document/mva2023/work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune/epoch_100.pth"
test_size = (3840, 2160)
video_name = "0002"

def load_cascade_rcnn(config_file, checkpoint_file, device="cuda"):
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

def visualize_detections(image, detections, threshold=0.3):
    """ 在影像上繪製檢測結果 """
    for i, det in enumerate(detections):
        if isinstance(det, torch.Tensor):
            det = det.cpu().numpy()

        for bbox in det:
            if bbox[4] > threshold:
                x1, y1, x2, y2, score = bbox
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Conf: {score:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def predict_videos(model, res_folder, args, video_filter=None):  # 加入 video_filter
    if osp.isdir(args.path):
        video_image_dict = get_video_image_dict(args.path)
    else:
        raise ValueError(f"args.path must be a directory, but got {args.path}")

    debug_output_dir = osp.join(res_folder, "debug_outputs")
    os.makedirs(debug_output_dir, exist_ok=True)

    # **只處理指定的影片資料夾**
    if video_filter and video_filter in video_image_dict:
        video_image_dict = {video_filter: video_image_dict[video_filter]}
    elif video_filter:
        raise ValueError(f"指定的影片 '{video_filter}' 不存在於 {args.path} 中")

    for video_name, files in video_image_dict.items():
        for frame_id, img_path in enumerate(files, 1):
            img = cv2.imread(img_path)
            detections = inference_detector(model, img)

            # 繪製檢測結果
            img_with_detections = visualize_detections(img, detections, threshold=args.track_thresh)

            # 儲存繪製後的圖片
            output_img_path = osp.join(debug_output_dir, f"{video_name}_{frame_id:04d}.jpg")
            cv2.imwrite(output_img_path, img_with_detections)

            if frame_id % 20 == 0:
                logger.info(f"Processed frame {frame_id} of {video_name}")

def main(args):
    if not args.path:
        raise ValueError("Please specify the path to the subset directory")

    output_dir = "Cascade_outputs/debug"  
    os.makedirs(output_dir, exist_ok=True)

    res_folder = osp.join(output_dir, Path(args.path).stem)
    os.makedirs(res_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    logger.info("Args: {}".format(args))

    model = load_cascade_rcnn(config_file, checkpoint_file, str(args.device))

    if args.fp16:
        model = model.half()

    print("Model loaded successfully!")

    predict_videos(model, res_folder, args, video_filter=video_name)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
