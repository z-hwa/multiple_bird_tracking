import os
import os.path as osp
from pathlib import Path

import torch
from loguru import logger

from tools.demo_track import Predictor
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser


def get_video_image_dict(root_path):
    '''get_video_image_dict(root_path)
    功能：讀取 root_path 目錄下的所有子目錄（代表不同影片），並獲取其中所有的影像檔路徑。

    流程

    遍歷 root_path 內的所有檔案與資料夾。
    過濾出 資料夾（視為一個影片）。
    遍歷該資料夾內的影像檔案（副檔名需符合 .jpg, .png 等）。
    按檔名排序後，建立 {影片名稱: [影像路徑列表]} 的 dict。'''
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

def predict_videos(predictor, res_folder, args):
    '''predict_videos(predictor, res_folder, args)
    功能：對 args.path 目錄內的所有影片影格 執行 YOLOX 物件偵測與 OCSort 追蹤，並存儲結果。

    流程

    讀取 args.path 內所有影片影格（get_video_image_dict(args.path)）。
    為每個影片 建立一個 OCSort 追蹤器。
    逐幀讀取影格：
    使用 predictor.inference(img_path, timer) 執行 YOLOX 偵測。
    若有偵測到物件：
    tracker.update() 更新追蹤結果。
    過濾掉太小或垂直比例過大的目標。
    記錄目標的 (frame_id, track_id, bbox資訊)。
    記錄處理時間 timer.toc() 來計算 FPS。
    儲存結果
    若 args.save_result，則將結果存入 res_folder/{影片名稱}.txt。'''

    if osp.isdir(args.path):
        video_image_dict = get_video_image_dict(args.path)
    else:
        raise ValueError(f"args.path must be a directory, but got {args.path}")

    for video_name, files in video_image_dict.items():
        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
        timer = Timer()
        results = []
        for frame_id, img_path in enumerate(files, 1):
            outputs, img_info = predictor.inference(img_path, timer)

            print('\n\n\n')
            print(type(outputs))
            print(outputs)
            print(args.track_thresh, args.iou_thresh, args.use_byte)
            print([img_info['height'], img_info['width']], exp.test_size)
            print('\n\n\n')
            input()

            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

                print(f"Frame {frame_id}:  {len(online_targets)} tracked objects")
                print(online_targets)
                input()

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
                timer.toc()
            else:
                timer.toc()

            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if args.save_result:
            res_file = osp.join(res_folder, f"{video_name}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

def main(exp, args):
    '''main(exp, args)
    功能：負責 初始化 YOLOX 模型、載入權重，並調用 predict_videos() 進行推論。

    流程

    確認參數：
    args.expn 設定實驗名稱。
    檢查 args.path 是否提供（必須是目錄）。
    建立 output_dir 來存儲結果。
    設定裝置
    args.device = "cuda" 若使用 GPU。
    model.to(args.device) 設定推論設備。
    載入模型
    exp.get_model() 取得 YOLOX 模型。
    model.load_state_dict(torch.load(ckpt_file)) 載入權重。
    若 args.fuse，則 fuse_model(model) 進行 BN 融合（加速推論）。
    若 args.fp16，則將模型轉換為 FP16（減少記憶體使用）。
    TensorRT 支援
    若 args.trt，則載入 TensorRT 優化後的 model_trt.pth。
    建立 Predictor 並開始推論
    predict_videos(predictor, res_folder, args)。'''

    if not args.expn:
        args.expn = exp.exp_name
    
    if not args.path:
        raise ValueError("Please specify the path to the subset directory")

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        res_folder = osp.join(output_dir, "predictions", Path(args.path).stem)
        os.makedirs(res_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    predict_videos(predictor, res_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
