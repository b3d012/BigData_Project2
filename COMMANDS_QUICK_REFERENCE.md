# Commands Quick Reference

## Environment creation

### Main environment
```powershell
conda create -n spark411 python=3.11 -y
conda activate spark411
pip install pandas numpy scikit-learn xgboost joblib openpyxl
```

### CICFlowMeter environment
```powershell
conda create -n cicflow python=3.12 -y
conda activate cicflow
pip install cicflowmeter==0.4.2
```

### Switch back to main environment
```powershell
conda activate spark411
```

---

## Tool checks

### Check Wireshark dumpcap
```powershell
& "C:\Program Files\Wireshark\dumpcap.exe" -D
```

### Add WinDump folder to PATH for current PowerShell session
```powershell
$env:Path += ";C:\Tools\WinDump"
where.exe tcpdump
where.exe windump
```

### Check CICFlowMeter
```powershell
conda activate cicflow
cicflowmeter -h
conda activate spark411
```

---

## Training commands

### Preprocess CIC-IDS2017
```powershell
python src/prepare_cicids2017.py
```

### Train baseline XGBoost
```powershell
python src/train_baseline_xgb.py
```

### Train stacking ensemble
```powershell
python src/train_stacking_ensemble.py
```

### Export confusion matrices
```powershell
python src/export_confusion_matrices.py
```

### Offline inference sanity check
```powershell
python src/predict_xgb_inference.py --input dataset/processed/cicids2017_binary_test.csv --name cic_test_inference
```

---

## Live commands

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

## Recommended run order from scratch

### Full rebuild
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

### Minimal rebuild
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

## Recommended Kali test scans

Use the host IP as the target.

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

Use one scan per 30-second window.
