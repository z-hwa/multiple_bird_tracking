import os
import json
import argparse

'''
為單純的影片資料集
建構COCO格式的空標註

修復bug
因為沒有添加類別導致 辨識以後輸出json會出錯 找不到對應的類別
'''

def create_coco_structure(root_dir, output_json):
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "bird"}]
    }
    image_id = 1

    # 遍歷影片資料夾 (0001, 0002, ...)
    for video_folder in sorted(os.listdir(root_dir)):
        video_path = os.path.join(root_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        # 遍歷每個影片資料夾內的圖片
        for img_file in sorted(os.listdir(video_path)):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                dataset["images"].append({
                    "id": image_id,
                    "file_name": f"{video_folder}/{img_file}",
                    "width": 3840,   # 這裡應該填入實際圖片尺寸
                    "height": 2160
                })
                image_id += 1

    # 儲存為 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    print(f"COCO JSON saved to {output_json}")

# root_dir = "/root/Document/data/MVA2025/split_pub_test"
# output_json = "/root/Document/data/MVA2025/annotations/split_test_coco_0030.json"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="為影片資料集建立 COCO 格式的空標註。")
    parser.add_argument("root_dir", help="影片資料集的根目錄路徑。")
    parser.add_argument("output_json", help="輸出的 COCO JSON 檔案路徑。")
    args = parser.parse_args()

    create_coco_structure(args.root_dir, args.output_json)