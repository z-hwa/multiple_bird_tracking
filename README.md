# MVA2025 submit

This is the step to reproduct the best inference result of zhwa2003.

## Overview

[TOC]

## 1. Enviroument requirement

The python version is 3.8 in my project.

We need to install following tools.
* mmdet
* hybrid sort

### i. mmdet

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

#### !!! May happened bug
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

### ii. Hybrid sort

To install Hybrid sort.
In the project's folder, run the following command.
```
cd HybridSORT
pip3 install -r requirements.txt
python3 setup.py develop
```

## 2. Model preparation

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

## 3. Inference

### i. prepare empty annotaion
To inference, we need prepare coco format empty annotation of the target data.

```
python MVA2025-SMOT4SB/scripts/convert_dir_to_coco.py <your_path_to_target_data_dir> <output_position_of_coco_format_empty_annotation>
```

---

#### Example
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

### ii. using mmdet to detection

We need using previous 2 model to detect bbox.

#### part 1.
First, for cascade_rcnn_swin_finetune_rfla_4stage.
Open the config file below.
```
MVA2023-SOD4SB/work_dirs/cascade_rcnn_swin_finetune_rfla_4stage/cascade_mask_rcnn_swin_finetune_rfla_4stage.py
```

Modify the data -> test -> anno_file, img_prefix attrs.
Replacing the original path by previous empty annotations path and img path.

---

##### Example
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

#### part 2.
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

### iii. Ensemble
Last, ensemble both detection result
```
python MVA2023-SOD4SB/tools/annotations/predict_result_ensemble_by_cut.py ./results_base.bbox.json ./results_cut.bbox.json ./result.bbox.json
```

## 4 Tracking

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
### example
Using phase2's data as example
```
sh MVA2025-SMOT4SB/scripts/predict_from_json_hybridSORT.sh --path /root/Document/data/MVA2025/pub_test
```

---

Finally, the mot format result will be output at
```
HybridSORT_outputs/private_test/
```
