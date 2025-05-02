import numpy as np
import os
import glob
import sys


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]  # 假設原始分數在第 7 列 (索引 6)
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h)
            f.write(line)


def dti(txt_path, save_path, n_min=25, n_dti=20):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        if seq_data.ndim == 1:
            seq_data = seq_data.reshape(1, -1)
        if seq_data.shape[1] != 9:
            print(f"警告: 文件 {seq_txt} 的列數不是 9，跳過。")
            continue

        min_id = int(np.min(seq_data[:, 1])) if seq_data.size > 0 else 0
        max_id = int(np.max(seq_data[:, 1])) if seq_data.size > 0 else 0
        seq_results = np.zeros((1, 9), dtype=np.float64)

        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]

            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 9), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, 1, 1] # 保持後面的兩個值為 1
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))

        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)

# test
# python3 OC_SORT/tools/my_tools/interpolation.py HybridSORT_outputs/phase_2/best_finetune_intersection/predictions/pub_test/ HybridSORT_outputs/phase_2_inter/best_finetune_intersection/predictions/pub_test/

# val
# python3 OC_SORT/tools/my_tools/interpolation.py ../HybridSORT/Cascade_outputs/val/best_TCM/predictions/val ../HybridSORT/Cascade_outputs/val_inter/best_TCM/predictions/val
if __name__ == '__main__':
    # txt_path, save_path, data_root, evaluation = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # evaluation = True if evaluation is 'True' else False
    # mkdir_if_missing(save_path)
    # dti(txt_path, save_path, n_min=30, n_dti=20)
    # if evaluation:
    #     print('Before DTI: ')
    #     eval_mota(data_root, txt_path)
    #     print('After DTI:')
    #     eval_mota(data_root, save_path)

    # txt_path, save_path = sys.argv[1], sys.argv[2]
    # data_root = 'datasets/mot/train'
    # mkdir_if_missing(save_path)
    # dti(txt_path, save_path, n_min=30, n_dti=20)
    # print('Before DTI: ')
    # eval_mota(data_root, txt_path)
    # print('After DTI:')
    # eval_mota(data_root, save_path)

    txt_path, save_path = sys.argv[1], sys.argv[2]
    mkdir_if_missing(save_path)
    dti(txt_path, save_path, n_min=30, n_dti=20)
    print('DTI 完成，結果保存在：', save_path)