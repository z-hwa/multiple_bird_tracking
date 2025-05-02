# 驗證 on smot4sb val

```
# 利用json預測 計算mot預測
sh scripts/predict_from_json_hybridSORT.sh --path /root/Document/data/SMOT4SB/val

# 轉換預測
python scripts/cp_preds_for_eval.py -i  Cascade_outputs/val_inter/best_finetune_5e5_old_ensemble_cut/predictions/val -o eval_inputs

# 轉換gt 只需要做一次!!!!!!!!
python3 scripts/oc_sort_ann_to_mot_ch.py -i /root/Document/data/SMOT4SB/annotations/val.json -o eval_inputs

# 計算分數
python3 TrackEval/scripts/run_smot4sb_challenge.py eval_inputs eval_outputs val --metric-smot4sb
```

# 生成比賽提交

```
# 利用json預測 計算mot預測
sh scripts/predict_from_json_hybridSORT.sh --path /root/Document/data/MVA2025/pub_test

生成提交檔案
python3 scripts/create_submission.py -i HybridSORT_outputs/phase_2/best_finetune_5e-5_sample_phase2/predictions/pub_test
```

# 生成插直

```
# test
python3 OC_SORT/tools/my_tools/interpolation.py HybridSORT_outputs/phase_2/best_finetune_intersection/predictions/pub_test/ HybridSORT_outputs/phase_2_inter/best_finetune_intersection/predictions/pub_test/


# val
python3 OC_SORT/tools/my_tools/interpolation.py Cascade_outputs/val/best_finetune_5e5_old_ensemble_cut/predictions/val Cascade_outputs/val_inter/best_finetune_5e5_old_ensemble_cut/predictions/val
```

# GP 插植
```
python OC_SORT/tools/my_tools/gp_interpolation.py HybridSORT_outputs/phase_2/best_finetune_intersection/predictions/pub_test HybridSORT_outputs/phase_2_inter/best_finetune_intersection/predictions/pub_test HybridSORT_outputs/phase_2_gp/best_finetune_intersection/predictions/pub_test
```

# ensembke

```
python EnsembleMOT/EnsembleMOT.py
```