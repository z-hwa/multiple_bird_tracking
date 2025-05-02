import numpy as np
import os
import glob
import sys

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,1\n' # 注意：這裡寫入的格式沒有包含 score
    with open(filename, 'w') as f:
        if results.ndim == 2 and results.shape[1] >= 6:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h)
                f.write(line)
        elif results.size == 0:
            pass # 寫入空檔案

def filter_top_k_percent(txt_path, save_path, k_percent=50):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    total_tracks_before_filter = 0
    total_tracks_after_filter = 0
    tracks_removed_count = 0

    mkdir_if_missing(save_path)

    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        if seq_data.ndim == 1:
            seq_data = seq_data.reshape(1, -1)
        if seq_data.shape[1] < 7: # 檢查列數是否至少為 7 (包含分數)
            print(f"警告: 文件 {seq_txt} 的列數少於 7，跳過。")
            continue

        track_scores = {}
        tracks_in_seq_before = set()

        for row in seq_data:
            track_id = int(row[1])
            tracks_in_seq_before.add(track_id)
            if track_id not in track_scores:
                track_scores[track_id] = {'total_score': 0, 'frame_count': 0}
            track_scores[track_id]['total_score'] += row[6]
            track_scores[track_id]['frame_count'] += 1
        total_tracks_before_filter += len(tracks_in_seq_before)

        avg_track_scores = {
            track_id: data['total_score'] / data['frame_count']
            for track_id, data in track_scores.items() if data['frame_count'] > 0
        }

        sorted_tracks = sorted(avg_track_scores.items(), key=lambda item: item[1], reverse=True)
        num_to_keep = int(len(sorted_tracks) * (k_percent / 100.0))
        top_k_tracks = [track_id for track_id, score in sorted_tracks[:num_to_keep]]

        filtered_results_for_seq = []
        tracks_in_seq_after = set()

        for row in seq_data:
            track_id = int(row[1])
            if track_id in top_k_tracks:
                filtered_results_for_seq.append(row)
                tracks_in_seq_after.add(track_id)

        filtered_results_np = np.array(filtered_results_for_seq) if filtered_results_for_seq else np.empty((0, 9))
        save_seq_txt = os.path.join(save_path, seq_name)
        write_results_score(save_seq_txt, filtered_results_np) # 總是寫入結果檔案

        total_tracks_after_filter += len(tracks_in_seq_after)
        tracks_removed_count += len(tracks_in_seq_before) - len(tracks_in_seq_after)

    print(f'保留平均分數前 {k_percent}% 的轨迹完成，結果保存在：', save_path)
    print(f'过滤前总轨迹数: {total_tracks_before_filter}')
    print(f'过滤后总轨迹数: {total_tracks_after_filter}')
    print(f'移除的轨迹数: {tracks_removed_count}')
    print(f'保留的轨迹数: {total_tracks_after_filter}')

if __name__ == '__main__':
    txt_path, save_path = sys.argv[1], sys.argv[2]
    k_percentage = 80 # 设置你想要保留的百分比
    filter_top_k_percent(txt_path, save_path, k_percentage)