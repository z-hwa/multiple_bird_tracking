#!/bin/bash

# 設定輸入和輸出目錄
PREDICTIONS_DIR="HybridSORT_outputs/phase_2_inter/best_finetune_intersection_demo/predictions/pub_test"
OUTPUT_DIR="../data/visualized_video_smot4sb_test_finetune_5e_with_old_intersection/"
IMAGES_DIR="/root/Document/data/MVA2025/pub_test"

# 確保輸出目錄存在
mkdir -p "$OUTPUT_DIR"

# 遍歷預測結果目錄中的所有 .txt 檔案
for PREDICTION_FILE in "$PREDICTIONS_DIR"/*.txt; do
  # 提取檔案名稱（不包含副檔名）
  FILENAME=$(basename "$PREDICTION_FILE" .txt)

  # 設定對應的圖片目錄
  IMAGE_SEQUENCE_DIR="$IMAGES_DIR/$FILENAME"

  # 執行視覺化指令
  python3 scripts/visualize_for_mot_ch.py \
    -m "$PREDICTION_FILE" \
    -o "$OUTPUT_DIR/$FILENAME" \
    -i "$IMAGE_SEQUENCE_DIR" \
    --mp4 \
    --show-bbox

  echo "已完成 $FILENAME 的視覺化"
done

echo "所有視覺化完成"

# 腳本說明：

# 設定目錄：
# PREDICTIONS_DIR：預測結果 .txt 檔案所在的目錄。
# OUTPUT_DIR：視覺化影片的輸出目錄。
# IMAGES_DIR：原始圖片序列所在的目錄。
# 創建輸出目錄：
# 使用 mkdir -p 確保輸出目錄存在，如果不存在則創建它。
# 遍歷預測結果：
# 使用 for 迴圈遍歷 PREDICTIONS_DIR 目錄中的所有 .txt 檔案。
# basename 指令用於提取檔案名稱（不包含副檔名）。
# 根據檔案名稱設定對應的圖片序列目錄 IMAGE_SEQUENCE_DIR。
# 執行視覺化指令：
# 使用您的 python3 scripts/visualize_for_mot_ch.py 指令，並將變數替換為當前處理的檔案和目錄。
# --mp4 和 --show-bbox 選項保持不變。
# 輸出完成訊息：
# 在每個檔案處理完成後，輸出一個訊息。
# 在所有檔案處理完成後，輸出一個總結訊息。
# 如何使用：

# 將上述程式碼複製並貼上到一個新的文字檔案中。
# 將檔案儲存為 visualize_all.sh（或其他您喜歡的名稱）。
# 在終端機中，使用 chmod +x visualize_all.sh 命令為該腳本添加執行權限。
# 執行腳本：./visualize_all.sh
# 注意事項：

# 請確保您的 python3 scripts/visualize_for_mot_ch.py 腳本和相關的 Python 環境已正確設定。
# 請根據您的實際目錄結構調整 PREDICTIONS_DIR、OUTPUT_DIR 和 IMAGES_DIR 變數。
# 如果您的圖片序列目錄中的圖片格式不是預設的，您可能需要在 visualize_for_mot_ch.py 腳本中進行相應的修改。
# 這個腳本將自動處理 pub_test 目錄中的所有 .txt 結果，並將視覺化影片儲存到 visualized_video 目錄中。