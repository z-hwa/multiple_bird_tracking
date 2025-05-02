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
 
def is_frequent_large_angle_turn_and_cross_screen(tracklet, width_threshold=0.8, height_threshold=0.8, angle_threshold_deg=90, min_large_angle_frames_ratio=0.1, turn_window=5):
    if len(tracklet) < 2 * turn_window + 1:
        return False

    large_angle_turn_count = 0
    for i in range(turn_window, len(tracklet) - turn_window):
        p_prev = tracklet[i - turn_window][2:4]
        p_curr = tracklet[i][2:4]
        p_next = tracklet[i + turn_window][2:4]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 > 1e-6 and norm_v2 > 1e-6:
            dot_product = np.dot(v1, v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)

            if angle_deg > angle_threshold_deg:
                large_angle_turn_count += 1

    turn_ratio = large_angle_turn_count / (len(tracklet) - 2 * turn_window) if (len(tracklet) - 2 * turn_window) > 0 else 0

    if turn_ratio >= min_large_angle_frames_ratio:
        min_x = np.min(tracklet[:, 2])
        max_x = np.max(tracklet[:, 2])
        min_y = np.min(tracklet[:, 3])
        max_y = np.max(tracklet[:, 3])

        width_span = max_x - min_x
        height_span = max_y - min_y  # 正確地定義 height_span

        estimated_width = 3840
        estimated_height = 2160

        if width_span > width_threshold * estimated_width and height_span > height_threshold * estimated_height:
            return True

    return False

def filter_tracks_by_motion(txt_path, save_path, width_threshold=0.8, height_threshold=0.8, angle_threshold_deg=90, min_large_angle_frames_ratio=0.1, turn_window=5):
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
        if seq_data.shape[1] < 7:
            print(f"警告: 文件 {seq_txt} 的列数少于 7，跳过。")
            continue

        min_id = int(np.min(seq_data[:, 1])) if seq_data.size > 0 else 0
        max_id = int(np.max(seq_data[:, 1])) if seq_data.size > 0 else 0
        filtered_results_for_seq = []
        tracks_in_seq_before = set()
        removed_tracks_in_seq = set()

        for row in seq_data:
            tracks_in_seq_before.add(int(row[1]))
        total_tracks_before_filter += len(tracks_in_seq_before)

        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            if len(tracklet) < 2 * turn_window + 1:
                filtered_results_for_seq.extend(tracklet)
                continue

            if not is_frequent_large_angle_turn_and_cross_screen(tracklet, width_threshold, height_threshold, angle_threshold_deg, min_large_angle_frames_ratio, turn_window):
                filtered_results_for_seq.extend(tracklet)
            else:
                removed_tracks_in_seq.add(track_id)

        filtered_results_np = np.array(filtered_results_for_seq) if filtered_results_for_seq else np.empty((0, 9))
        tracks_in_seq_after = set(filtered_results_np[:, 1].astype(int)) if filtered_results_np.size > 0 else set()

        save_seq_txt = os.path.join(save_path, seq_name)
        write_results_score(save_seq_txt, filtered_results_np)

        total_tracks_after_filter += len(tracks_in_seq_after)
        tracks_removed_count += len(tracks_in_seq_before) - len(tracks_in_seq_after)

    print('基于大角度转弯和画面覆盖的轨迹过滤完成，结果保存在：', save_path)
    print(f'过滤前总轨迹数: {total_tracks_before_filter}')
    print(f'过滤后总轨迹数: {total_tracks_after_filter}')
    print(f'移除的轨迹数: {tracks_removed_count}')
    print(f'保留的轨迹数: {total_tracks_after_filter}')

if __name__ == '__main__':
    txt_path, save_path = sys.argv[1], sys.argv[2]
    width_threshold = 0.5
    height_threshold = 0.5
    angle_threshold_deg = 30  # 更低的阈值来捕捉大的转弯
    min_large_angle_frames_ratio = 0.05
    turn_window = 10          # 考虑前后 10 帧的运动

    filter_tracks_by_motion(txt_path, save_path, width_threshold, height_threshold, angle_threshold_deg, min_large_angle_frames_ratio, turn_window)
    
# python scripts/tools/track_bent_filter.py HybridSORT_outputs/phase_2/best_tcm/predictions/pub_test HybridSORT_outputs/phase_2_track_filter/best_tcm/predictions/pub_test
