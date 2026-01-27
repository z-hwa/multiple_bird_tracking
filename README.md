# Intersection-based Ensemble for Small Multi-Object Tracking

本專案實作了一種專為複雜環境（如都市電線桿場景）設計的**小目標多物件追蹤 (Small MOT)** 系統。本研究論文已被 **ICMVA 2025** 接收，並在 MVA2025 SMOT4SB 挑戰賽中獲得 **Private Leaderboard 第 2 名** 的成績。

## 專案亮點
* **Intersection-based Ensemble (IE)**：提出一種簡單且極其有效的集成策略，利用兩個互補的檢測模型大幅過濾偽陽性（False Positives）。
* **兩階段檢測架構**：結合強大的基礎模型 (RB Model) 與針對特定挑戰場景微調的細化模型 (RS Model)。
* **高效追蹤集成**：採用 Hybrid-SORT 結合 DIoU 關聯度量，無需額外訓練 Re-ID 模型即可達成穩定的追蹤效果。

## 核心改進：Intersection-based Ensemble (IE)

在都市環境中，電線桿上的絕緣子（Insulators）外型極其類似小鳥，常導致傳統檢測器產生大量誤判。

### 改進原理
1. **RB Model**：在廣泛的鳥類數據集上訓練，具備高召喚率但容易誤判絕緣子。
2. **RS Model**：針對電線桿場景微調，對特定誤判點有較佳的判別力。
3. **IE 策略**：只有當兩個模型同時在同一位置預測出目標（IoU > 0）時，才保留檢測結果。這成功過濾了僅由單一模型產生的偽陽性。

### 改進前後視覺化對比
| 改進前 (RB Model Only) | 改進後 (After IE Strategy) |
| :--- | :--- |
| ![Before IE](https://github.com/z-hwa/multiple_bird_tracking/blob/main/docs/before_ie.png?raw=true) | ![After IE](https://github.com/z-hwa/multiple_bird_tracking/blob/main/docs/after_ie.png?raw=true) |
| *電線上充滿了由絕緣子引起的錯誤追蹤軌跡* | *僅保留真正的小鳥追蹤結果（紅框處）* |

## 實驗結果

在 MVA2025 Challenge 的測試集上，IE 策略顯著提升了精確度：

| 方法 | SO-HOTA | SO-DetPr (精確度) | CLR FP (誤判數) |
| :--- | :--- | :--- | :--- |
| 原始 RB 模型 | 42.96 | 57.29 | 19,091 |
| **IE 集成策略** | **47.13** | **77.88** (+20.6) | **7,629** (-11,462) |
| **IE + 線性內插** | **49.20** | - | - |

* 最終系統透過線性內插（Linear Interpolation）進一步優化軌跡連續性，達到 **49.20** 的 SO-HOTA 高分。

## 技術細節
* **Detector**: Cascade R-CNN (Backbone: Swin Transformer)
* **Label Assignment**: RFLA (Gaussian Receptive Field based)
* **Tracker**: Hybrid-SORT with DIoU Association
* **Data Augmentation**: Copy-Paste augmentation with Kaggle "Birds Flying" dataset


## How to run this Repo

This is the step to reproduct the best inference result of zhwa2003.
To reproduct, following the step from 1 to 5.

### 1. Enviroument requirement

The python version is 3.8 in my project.

We need to install following tools.
* mmdet
* hybrid sort

#### i. mmdet

To install mmdet, we need install cuda and pytorch first.
```
In my work, using cuda 12.1 and pytorch 2.4.1
```

them, we need mmcv-full 1.7.2
```
pip install -U openmim
mim install mmcv-full==1.7.2
```

Last, install the mmdet.
In the project's folder, run the following command
```
cd MVA2023-SOD4SB
pip install -v -e .
```

##### !!! May happened bug
1. There is no module timm:
    ```
    pip install timm
    ```
2. mmcv's device detect problem: 
    ref: https://github.com/open-mmlab/mmdetection/issues/10720

    Find the code in the first image(which located at "mmcv\parallel\\_functions.py")
    Then replace by the second image's code.
    ![image](https://hackmd.io/_uploads/H1cKZ4Z0kx.png)
    ![image](https://hackmd.io/_uploads/S1x9bNWAye.png)

   you can also copy from here
   ```
   tmp_target_gpus = [torch.device('cuda', device) for device in target_gpus]
   streams = [_get_stream(device) for device in tmp_target_gpus]
   ```

#### ii. Hybrid sort

To install Hybrid sort.
In the project's folder, run the following command.
```
cd HybridSORT
pip3 install -r requirements.txt
python3 setup.py develop
```

### 2. Model preparation

Download model from following link.
https://drive.google.com/drive/folders/1bpvrOTGZTS2sOmRMLifXgGd3prs1UukZ?usp=sharing

There are two model folder.
Each contains a model.
```
cascade_rcnn_swin_finetune_rfla_4stage
cascade_rcnn_swin_rfla_smot4sb
```

Download these two model.
Put them into the corresponding project folder.

1. For the model in cascade_rcnn_swin_finetune_rfla_4stage
    ```
    MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_finetune_rfla_4stage
    ```
2. For the model in cascade_rcnn_swin_rfla_smot4sb
    ```
    MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_rfla_smot4sb
    ```
    
Finally, the dir structure will be look like,
```
MVA2023-SOD4SB/work_dirs/
├── cascade_rcnn_swin_finetune_rfla_4stage
│   └── cascade_mask_rcnn_swin_finetune_rfla_4stage.py
└── cascade_rcnn_swin_rfla_smot4sb
    └── cascade_rcnn_swin_rfla_smot4sb.py

2 directories, 2 files
```

### 3. Inference

#### i. prepare empty annotaion
To inference, we need prepare coco format empty annotation of the target data.

```
python MVA2025-SMOT4SB/scripts/convert_dir_to_coco.py <your_path_to_target_data_dir> <output_position_of_coco_format_empty_annotation>
```

---

##### Example
Using phase2's data as example, we put public data at
```
/root/Document/data/MVA2025/pub_test
```

the dir structure will be like
```
pub_test/
├── 0001
├── 0002
├── 0003
├── 0004
...
├── 0036
├── 0037
└── 0038
```

Thus, run the following code
```
python MVA2025-SMOT4SB/scripts/convert_dir_to_coco.py /root/Document/data/MVA2025/pub_test ./test_coco.json
```

---

#### ii. using mmdet to detection

We need using previous 2 model to detect bbox.

##### part 1.
First, for cascade_rcnn_swin_finetune_rfla_4stage.
Open the config file below.
```
MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_finetune_rfla_4stage/cascade_mask_rcnn_swin_finetune_rfla_4stage.py
```

Modify the data -> test -> anno_file, img_prefix attrs.
Replacing the original path by previous empty annotations path and img path.

---

###### Example
Using phase2's data as example, the final result will lokk like
![Untitled](https://hackmd.io/_uploads/H1LdRxGxxx.png)

---

Then run the following command,
```
bash MVA2023-SOD4SB/tools/test.sh MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_finetune_rfla_4stage/cascade_mask_rcnn_swin_finetune_rfla_4stage.py MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_finetune_rfla_4stage/epoch_104.pth --format-only --eval-options jsonfile_prefix=results_base
```

After that, we will get the first detection result
```
results_base.bbox.json
```

##### part 2.
Second, we do the same thing for other model.
Modifing the following config file first.
```
MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_rfla_smot4sb/cascade_rcnn_swin_rfla_smot4sb.py
```

Then, detection.
```
bash MVA2023-SOD4SB/tools/test.sh MVA2023-SOD4SB/work_dirs/ccascade_rcnn_swin_rfla_smot4sb/cascade_rcnn_swin_rfla_smot4sb.py MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_rfla_smot4sb/epoch_5.pth --format-only --eval-options jsonfile_prefix=results_cut
```

After that, we will get the second detection result
```
results_cut.bbox.json
```

#### iii. Ensemble
Last, ensemble both detection result.
```
python MVA2023-SOD4SB/tools/annotations/predict_result_ensemble_by_cut.py ./results_base.bbox.json ./results_cut.bbox.json ./result.bbox.json
```

### 4 Tracking

First, open the file
```
MVA2025-SMOT4SB/OC_SORT/tools/predict_from_json_hybrid_sort.py
```

Then, modify the output_dir, predictions_json_path, annotations_json_path attrs as follow.
```pyt=
output_dir = "HybridSORT_outputs/private_test"  # result output path
predictions_json_path = "./result.bbox.json" # previous result prediction
annotations_json_path = "./test_coco.json" # empty result prediction
```

using previous final prediction result, to track
```
sh MVA2025-SMOT4SB/scripts/predict_from_json_hybridSORT.sh --path <path_of_your_target_dataset>
```

---
#### example
Using phase2's data as example
```
sh MVA2025-SMOT4SB/scripts/predict_from_json_hybridSORT.sh --path /root/Document/data/MVA2025/pub_test
```

---

Finally, the mot format result will be output at
```
HybridSORT_outputs/private_test/
```

### 5 Post Processing

After we got the mot format result, we do the postprocessing next.

So we do interpolation by following code.
```
python MVA2025-SMOT4SB/OC_SORT/tools/my_tools/interpolation.py <previous_mot_result_folder> <final_output_path>
```

---

#### Example

```
python3 MVA2025-SMOT4SB/OC_SORT/tools/my_tools/interpolation.py HybridSORT_outputs/private_test/predictions/pub_test/ HybridSORT_outputs/private_test_inter/predictions/pub_test/
```

After upper code working, the final result will be output at folder "HybridSORT_outputs/private_test_inter"


---

### 論文引用
如果您覺得本專案對您的研究有幫助，請引用我們的論文：
> Guan-Zhang Wang and Wei-Ta Chu, "Intersection-based Ensemble for Small Multi-Object Tracking in Challenging Environments," 2025 19th International Conference on Machine Vision and Applications (MVA), Kyoto, Japan, 2025, pp. 1-6, doi: 10.23919/MVA65244.2025.11175071.,



### 作者
* **王冠章 (Guan-Zhang Wang)** - 國立成功大學資訊工程學系
