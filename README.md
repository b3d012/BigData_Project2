# BigData_Project2
Chatgpt No Codex


[INFO] Found 8 CSV files
[LOAD] Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[LOAD] Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[LOAD] Friday-WorkingHours-Morning.pcap_ISCX.csv
[LOAD] Monday-WorkingHours.pcap_ISCX.csv
[LOAD] Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[LOAD] Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[LOAD] Tuesday-WorkingHours.pcap_ISCX.csv
[LOAD] Wednesday-workingHours.pcap_ISCX.csv
[INFO] Merged shape: (2830743, 80)
[INFO] Dropped exact duplicates: 256479
[INFO] Missing feature values before imputation: 3248
[INFO] Missing feature values after imputation: 0
[SAVE] dataset\processed\cicids2017_binary_full.csv
[SAVE] dataset\processed\cicids2017_binary_train.csv
[SAVE] dataset\processed\cicids2017_binary_val.csv
[SAVE] dataset\processed\cicids2017_binary_test.csv
[SAVE] dataset\processed\feature_columns_binary.json
[SAVE] dataset\processed\label_mapping_binary.json
[SAVE] dataset\processed\median_imputer_binary.json
[SAVE] dataset\processed\preprocess_summary_binary.json

[DONE] Preprocessing complete.
[INFO] Final feature count: 70
[INFO] Final classes: ['ATTACK', 'BENIGN']

------------------------------------------------------

[INFO] Generating validation probabilities...
[INFO] Best validation threshold for ATTACK: 0.891280472278595
[INFO] Validation F1 at best threshold: 0.9977233075412503
[INFO] Evaluating validation split...
[INFO] Evaluating test split...
[INFO] Saving model and artifacts...

[DONE] XGBoost baseline training complete.
[SAVE] models\baseline_xgb\xgb_baseline_model.json
[SAVE] models\baseline_xgb\xgb_baseline_model.joblib
[SAVE] models\baseline_xgb\selected_threshold.json
[SAVE] models\baseline_xgb\val_metrics.json
[SAVE] models\baseline_xgb\test_metrics.json
[SAVE] models\baseline_xgb\inference_schema.json

[SUMMARY]
Validation F1 (ATTACK): 0.997723
Validation Recall (ATTACK): 0.998137
Validation Precision (ATTACK): 0.99731
Validation ROC-AUC (ATTACK): 0.999979
Validation AP (ATTACK): 0.999897
Test F1 (ATTACK): 0.997497
Test Recall (ATTACK): 0.998028
Test Precision (ATTACK): 0.996966
Test ROC-AUC (ATTACK): 0.99998
Test AP (ATTACK): 0.999897
(spark411) PS C:\Users\abdul\Desktop\Project\BigData_Project2> 