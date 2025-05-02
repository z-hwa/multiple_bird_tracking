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
        if results.ndim == 2 and results.shape[1] >= 6:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h)
                f.write(line)

def filter_tracks_by_score(txt_path, save_path, score_threshold=0.3, min_track_length=10): # 添加 min_track_length 参数
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    total_tracks_before_filter = 0
    total_tracks_after_filter = 0
    tracks_removed_count = 0
    removed_tracks_by_sequence = {} # 存储每个序列中被移除的轨迹 ID

    mkdir_if_missing(save_path)

    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        if seq_data.ndim == 1:
            seq_data = seq_data.reshape(1, -1)
        if seq_data.shape[1] < 7:
            print(f"警告: 文件 {seq_txt} 的列数少于 7，跳过。")
            continue

        min_id = int(np.min(seq_data[:, 1])) if seq_data.size > 0 else 0
        max_id = int(np.max(seq_data[:, 1])) if seq_data.size > 0 else 0
        filtered_results_for_seq = []
        track_scores = {} # 存储每条轨迹的总分数和帧数
        tracks_in_seq_before = set()
        removed_tracks_in_current_seq = set() # 存储当前序列中被移除的轨迹 ID

        for row in seq_data:
            tracks_in_seq_before.add(int(row[1]))
        total_tracks_before_filter += len(tracks_in_seq_before)

        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            if tracklet.shape[0] == 0:
                continue

            n_frame = tracklet.shape[0]
            total_score = 0
            for row in tracklet:
                total_score += row[6] # 假设分数在第 7 列 (索引 6)

            avg_score = total_score / n_frame if n_frame > 0 else 0
            track_scores[track_id] = avg_score

            # 只对长度超过 min_track_length 的轨迹进行分数过滤
            if n_frame > min_track_length:
                if avg_score < score_threshold:
                    removed_tracks_in_current_seq.add(track_id)
                else:
                    filtered_results_for_seq.extend(tracklet)
            else:
                # 短轨迹直接保留
                filtered_results_for_seq.extend(tracklet)

        filtered_results_np = np.array(filtered_results_for_seq) if filtered_results_for_seq else np.empty((0, 9))
        tracks_in_seq_after = set(filtered_results_np[:, 1].astype(int)) if filtered_results_np.size > 0 else set()

        save_seq_txt = os.path.join(save_path, seq_name)
        write_results_score(save_seq_txt, filtered_results_np)

        total_tracks_after_filter += len(tracks_in_seq_after)
        tracks_removed_count += len(tracks_in_seq_before) - len(tracks_in_seq_after)
        removed_tracks_by_sequence[seq_name] = sorted(list(removed_tracks_in_current_seq))

    print('低分轨迹过滤完成，結果保存在：', save_path)
    print(f'过滤前总轨迹数: {total_tracks_before_filter}')
    print(f'过滤后总轨迹数: {total_tracks_after_filter}')
    print(f'移除的轨迹总数: {tracks_removed_count}')
    print(f'保留的轨迹总数: {total_tracks_after_filter}')

    print('\n被移除的轨迹 (按影片):')
    for seq_name, removed_ids in removed_tracks_by_sequence.items():
        if removed_ids:
            print(f'  {seq_name}: {removed_ids}')

if __name__ == '__main__':
    txt_path, save_path = sys.argv[1], sys.argv[2]
    filter_tracks_by_score(txt_path, save_path, score_threshold=0.5, min_track_length=200) # 设置你的分数阈值和最小轨迹长度
    
# python scripts/tools/track_filter.py HybridSORT_outputs/phase_2_inter/best_tcm/predictions/pub_test HybridSORT_outputs/phase_2_track_filter/best_tcm/predictions/pub_test
