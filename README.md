# BigData Project 2

## Overview

This project builds a binary intrusion detection pipeline using CIC-IDS2017 and then uses the trained XGBoost model for live traffic classification.

The full project supports:

1. preprocessing the original CIC-IDS2017 dataset
2. training the baseline XGBoost model
3. optionally training the stacking ensemble
4. optionally exporting confusion matrices for all trained models
5. optionally running an offline inference sanity check
6. running the final live pipeline:
   - capture PCAP windows with `dumpcap`
   - convert PCAP to flow CSV using `cicflowmeter`
   - adapt the flow CSV to the saved training schema
   - run the trained XGBoost model
   - append the result to Excel

The final chosen deployment model is the standalone XGBoost baseline.

---

## Main pipeline

Training pipeline:

1. put the original CIC-IDS2017 CSV files into `dataset/raw/`
2. preprocess and split the dataset
3. train the baseline XGBoost model
4. optionally train the stacking ensemble
5. optionally export confusion matrices

Live pipeline:

1. capture a fresh PCAP window with `dumpcap`
2. convert the PCAP into a CICFlowMeter CSV
3. adapt the CICFlowMeter CSV column names to the training schema
4. apply the saved baseline XGBoost model
5. append the window result to Excel

---

## Project structure

```text
project/
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── test1/
├── live_capture/
├── models/
├── outputs/
├── src/
└── README.md
```

Recommended structure after setup:

```text
project/
├── dataset/
│   ├── raw/                  # original CIC-IDS2017 MachineLearningCSV files
│   ├── processed/            # processed train/val/test splits and schema files
│   └── test1/                # optional extra test datasets
├── live_capture/
│   ├── pcaps/
│   ├── flows_raw/
│   ├── flows_model_ready/
│   ├── predictions/
│   └── summaries/
├── models/
│   ├── baseline_xgb/
│   └── stacking_ensemble/
├── outputs/
│   ├── inference/
│   ├── live_attack_results.xlsx
│   └── confusion_matrices_summary.json
├── src/
│   ├── prepare_cicids2017.py
│   ├── train_baseline_xgb.py
│   ├── train_stacking_ensemble.py
│   ├── export_confusion_matrices.py
│   ├── predict_xgb_inference.py
│   └── live_capture_cicflow_xgb_excel.py
└── README.md
```

---

## Final model choice

The final chosen model is the standalone XGBoost baseline.

Reason:
- it outperformed the stacking ensemble overall on CIC-IDS2017
- it is simpler to deploy
- it gave extremely strong validation and test performance
- it is the model used in the final live pipeline

The stacking ensemble is still included for:
- comparison
- reporting
- confusion matrix generation
- analysis of the base models inside the stack

---

## Software requirements

## 1. Conda / Python

Use Conda to create the environments.

You will use two environments:

- `spark411`
  - preprocessing
  - training
  - ensemble
  - confusion matrix export
  - offline inference
  - final live script

- `cicflow`
  - `cicflowmeter`

---

## 2. Wireshark

Install Wireshark so `dumpcap.exe` is available.

Typical path on Windows:

```text
C:\Program Files\Wireshark\dumpcap.exe
```

Check that it works:

```powershell
& "C:\Program Files\Wireshark\dumpcap.exe" -D
```

This lists the available interfaces. Use the correct interface number later in the live script.

---

## 3. WinDump / tcpdump

The Python `cicflowmeter` package needs a `tcpdump`-style executable on Windows.

Put these in a folder such as:

```text
C:\Tools\WinDump\
```

Required files:
- `windump.exe`
- `tcpdump.exe`

If needed, `tcpdump.exe` can just be a copy of `windump.exe`.

For the current PowerShell session, add it to PATH:

```powershell
$env:Path += ";C:\Tools\WinDump"
where.exe tcpdump
where.exe windump
```

Both commands should return a valid path.

---

## 4. Original CIC-IDS2017 dataset

Download the `MachineLearningCSV.zip` version and extract the 8 CSV files into:

```text
dataset/raw/
```

Expected files:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

Only the original training dataset is required for the minimal rebuild path.

---

## Environment setup

## A. Main environment

Use this for:
- preprocessing
- baseline training
- ensemble training
- confusion matrix export
- offline inference
- final live capture script

```powershell
conda create -n spark411 python=3.11 -y
conda activate spark411
pip install pandas numpy scikit-learn xgboost joblib openpyxl
```

---

## B. CICFlowMeter environment

Use this only for `cicflowmeter`.

```powershell
conda create -n cicflow python=3.12 -y
conda activate cicflow
pip install cicflowmeter==0.4.2
```

Check that it responds:

```powershell
cicflowmeter -h
```

Then switch back to the main environment when needed:

```powershell
conda activate spark411
```

---

## Recommended Windows command checks

Check that Wireshark exists:

```powershell
Test-Path "C:\Program Files\Wireshark\dumpcap.exe"
```

Check that WinDump is discoverable:

```powershell
where.exe tcpdump
where.exe windump
```

Check that CICFlowMeter exists:

```powershell
Test-Path "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe"
```

Replace `YOUR_USERNAME` with the correct Windows username.

---

## Training pipeline from scratch

## 1. Activate the main environment

```powershell
conda activate spark411
```

## 2. Preprocess CIC-IDS2017

This script:
- loads the original 8 CSV files
- merges them
- cleans column names
- removes repeated header rows
- handles NaN / Infinity
- creates binary labels
- freezes the final 70-feature schema
- saves train/val/test splits

Run:

```powershell
python src/prepare_cicids2017.py
```

Important outputs:

- `dataset/processed/cicids2017_binary_full.csv`
- `dataset/processed/cicids2017_binary_train.csv`
- `dataset/processed/cicids2017_binary_val.csv`
- `dataset/processed/cicids2017_binary_test.csv`
- `dataset/processed/feature_columns_binary.json`
- `dataset/processed/label_mapping_binary.json`
- `dataset/processed/median_imputer_binary.json`
- `dataset/processed/preprocess_summary_binary.json`

---

## 3. Train the baseline XGBoost model

This is the final chosen model.

Run:

```powershell
python src/train_baseline_xgb.py
```

Important outputs:

- `models/baseline_xgb/xgb_baseline_model.joblib`
- `models/baseline_xgb/xgb_baseline_model.json`
- `models/baseline_xgb/selected_threshold.json`
- `models/baseline_xgb/val_metrics.json`
- `models/baseline_xgb/test_metrics.json`
- `models/baseline_xgb/inference_schema.json`

---

## Optional ensemble pipeline

## 4. Train the stacking ensemble

Base models:
- Random Forest
- HistGradientBoosting
- Logistic Regression
- XGBoost

Meta-learner:
- Logistic Regression

Run:

```powershell
python src/train_stacking_ensemble.py
```

Important outputs:

- `models/stacking_ensemble/stack_val_metrics.json`
- `models/stacking_ensemble/stack_test_metrics.json`
- `models/stacking_ensemble/base_model_metrics.json`
- `models/stacking_ensemble/stack_threshold_info.json`
- `models/stacking_ensemble/oof_meta_train.csv`
- `models/stacking_ensemble/meta_val.csv`
- `models/stacking_ensemble/meta_test.csv`

---

## Optional confusion matrix export

## 5. Export confusion matrices for all trained models

This includes:
- baseline XGBoost
- final stacked ensemble
- base models inside the stack

Run:

```powershell
python src/export_confusion_matrices.py
```

Important output:

- `outputs/confusion_matrices_summary.json`

---


## Why the live pipeline uses dumpcap

The final live pipeline uses:

**dumpcap -> CICFlowMeter -> adapter -> model -> Excel**

Reason:
- `dumpcap` is the clean raw capture tool
- it reliably saves PCAP windows
- the PCAP can then be passed into CICFlowMeter
- this is closer to the original training data style than a direct tshark-only flow reconstruction

This also gives:
- reproducibility
- stored PCAP evidence
- the ability to re-run CICFlowMeter or re-test the model later on the same capture

---

## Live pipeline architecture

The final script does this for each window:

1. delete old contents in `live_capture/`
2. capture a fresh PCAP window with `dumpcap`
3. convert the PCAP to a CICFlowMeter CSV
4. adapt the CICFlowMeter CSV column names to the training schema
5. align to the saved 70-feature schema
6. fill missing values with the saved training medians
7. apply the trained baseline XGBoost model
8. classify the whole window
9. append the result to Excel

The first two important Excel columns are:
- `Attack itself`
- `Classification`

---

## Live pipeline prerequisites

Before using the final live script, make sure all of these are true:

### A. `dumpcap` works

```powershell
& "C:\Program Files\Wireshark\dumpcap.exe" -D
```

### B. `cicflowmeter` works

Activate the CICFlowMeter environment:

```powershell
conda activate cicflow
cicflowmeter -h
```

Then return to the main environment:

```powershell
conda activate spark411
```

### C. `tcpdump.exe` is on PATH for the PowerShell session

```powershell
$env:Path += ";C:\Tools\WinDump"
where.exe tcpdump
where.exe windump
```

### D. The baseline XGBoost model already exists

Required:
- `models/baseline_xgb/xgb_baseline_model.joblib`
- `models/baseline_xgb/selected_threshold.json`

### E. The processed schema files already exist

Required:
- `dataset/processed/feature_columns_binary.json`
- `dataset/processed/label_mapping_binary.json`
- `dataset/processed/median_imputer_binary.json`

---

## Final live script

## 7. One-window live test

This captures one 30-second window, processes it, writes the outputs, appends to Excel, and stops.

```powershell
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:/Program Files/Wireshark/dumpcap.exe" `
  --cicflowmeter-exe "C:/Users/YOUR_USERNAME/miniconda3/envs/cicflow/Scripts/cicflowmeter.exe"`
  --window-seconds 5 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1 `
  --max-windows 1
```

Replace:
- `5` with the correct interface number
- `YOUR_USERNAME` with the correct Windows username

python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\abdul\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1 `
  --max-windows 5
---

## 8. Continuous live monitoring

This keeps processing new windows until you stop it.

```powershell
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1
```

---

## Live outputs

The live script creates:

- `live_capture/pcaps/`
- `live_capture/flows_raw/`
- `live_capture/flows_model_ready/`
- `live_capture/predictions/`
- `live_capture/summaries/`
- `outputs/live_attack_results.xlsx`

The script clears old contents inside `live_capture/` at the start of each run.

The Excel file is preserved and appended to.

---

## What a successful live run means

A successful live run means all of these happened:
- old `live_capture` contents were deleted
- a fresh PCAP window was captured
- CICFlowMeter produced a raw flow CSV
- the adapter converted the CSV to the model-ready schema
- the baseline XGBoost model ran
- the window result was appended to Excel

If you captured only normal traffic, the result may be `BENIGN`, which is expected.

---

## Recommended test scenarios

These are recommended scenarios for checking the final live pipeline.

## Scenario 1. Normal traffic baseline

Goal:
- confirm the pipeline works
- confirm regular traffic usually becomes `BENIGN`

Run the one-window live script and do nothing unusual during the 30-second window.

Expected:
- `Classification = BENIGN`
- `Flows predicted ATTACK = 0` or very low

---

## Scenario 2. Kali Linux attack window

Goal:
- run the final live script
- send traffic from Kali during the capture window
- see whether the window becomes `ATTACK`

Procedure:
1. start the live script on the host
2. during the 30-second window, launch one Kali scan
3. let the window finish
4. inspect:
   - Excel
   - prediction CSV
   - summary JSON

Use one scan per window so the result is easier to interpret.

---

## Recommended Nmap tests from Kali

Use the host machine IP as the target.

Recommended order:

### Window 1
```bash
nmap -sn <host_ip>
```
### Window 2
```bash
nmap -sS <host_ip>
```
### Window 3
```bash
nmap -Pn -sS -p 1-1000 <host_ip>
```
### Window 4
```bash
nmap -sV <host_ip>
```
### Window 5
```bash
nmap -A <host_ip>
```


Recommended practice:
- do one Nmap type per 30-second window
- start the scan about 3 to 5 seconds after the capture begins

Do **not** run all 5 scans in the same 30-second window if you want clear results.

---

## Recommended VirtualBox networking setup for Kali

Best lab setup:
- **Adapter 1 = NAT** for Kali internet access
- **Adapter 2 = Host-Only Adapter** for the private host-to-VM lab path

If host-only is broken or unavailable:
- use **Bridged Adapter** temporarily as a fallback

If host-only networking fails because the host-only driver is missing:
- repair or reinstall VirtualBox
- reboot
- create the host-only interface again

---

## Recommended run orders

## Full rebuild from scratch

Use this if you are rebuilding the project on a new device and want everything again, including the ensemble and confusion matrices.

```powershell
conda activate spark411
python src/prepare_cicids2017.py
python src/train_baseline_xgb.py
python src/train_stacking_ensemble.py
python src/export_confusion_matrices.py
python src/predict_xgb_inference.py --input dataset/processed/cicids2017_binary_test.csv --name cic_test_inference
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1
```

---

## Minimal rebuild from scratch

Use this if you only care about the final chosen model and the live pipeline.

```powershell
conda activate spark411
python src/prepare_cicids2017.py
python src/train_baseline_xgb.py
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1
```

---

## Notes on CICFlowMeter compatibility

The live pipeline is **compatible** with the training pipeline, but it is not guaranteed to be **byte-for-byte identical** to the original official CIC-IDS2017 generation process.

What matches:
- same type of feature data
- same inference schema
- same saved 70-feature order
- same saved training medians
- same saved XGBoost model
- same saved threshold

What is not guaranteed to be identical:
- exact CICFlowMeter version / implementation
- exact column naming
- exact tool defaults or flow-generation settings

This is why the adapter and the saved inference schema are important.

---

## Notes on the stacking ensemble

The stacking ensemble is included because:
- it was implemented successfully
- it can be re-trained
- confusion matrices can be exported for all base models and the final stack

But the final deployment choice is still the standalone XGBoost model.

---

## Optional commands summary

### Preprocess
```powershell
python src/prepare_cicids2017.py
```

### Baseline XGBoost
```powershell
python src/train_baseline_xgb.py
```

### Stacking ensemble
```powershell
python src/train_stacking_ensemble.py
```

### Confusion matrices
```powershell
python src/export_confusion_matrices.py
```

### Offline inference sanity check
```powershell
python src/predict_xgb_inference.py --input dataset/processed/cicids2017_binary_test.csv --name cic_test_inference
```

### One-window live test
```powershell
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1 `
  --max-windows 1
```

### Continuous live monitoring
```powershell
python src/live_capture_cicflow_xgb_excel.py `
  --interface 5 `
  --dumpcap-exe "C:\Program Files\Wireshark\dumpcap.exe" `
  --cicflowmeter-exe "C:\Users\YOUR_USERNAME\miniconda3\envs\cicflow\Scripts\cicflowmeter.exe" `
  --window-seconds 30 `
  --attack-label "Kali Live Test" `
  --excel outputs\live_attack_results.xlsx `
  --min-attack-flows 1
```

---

## Quick practical checklist

Before the live demo, make sure:

- [ ] `dataset/raw/` contains the original 8 CIC-IDS2017 CSV files
- [ ] `python src/prepare_cicids2017.py` completed
- [ ] `python src/train_baseline_xgb.py` completed
- [ ] `models/baseline_xgb/xgb_baseline_model.joblib` exists
- [ ] `models/baseline_xgb/selected_threshold.json` exists
- [ ] `dataset/processed/feature_columns_binary.json` exists
- [ ] `dataset/processed/label_mapping_binary.json` exists
- [ ] `dataset/processed/median_imputer_binary.json` exists
- [ ] `dumpcap.exe` works
- [ ] `cicflowmeter -h` works
- [ ] `where.exe tcpdump` returns a path
- [ ] the correct interface number is known
- [ ] the Kali VM can reach the host target IP

---

## Final note

For the final presentation or demo, the simplest defensible story is:

1. train the binary model on CIC-IDS2017
2. choose standalone XGBoost as the final model
3. optionally report the stack and confusion matrices for comparison
4. capture live traffic into PCAP windows
5. convert those windows to CICFlowMeter CSVs
6. adapt the CSVs to the saved training schema
7. run the baseline XGBoost model
8. save the result to Excel
