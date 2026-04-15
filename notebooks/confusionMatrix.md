Baseline XGBoost:
[[63756, 126], [194, 322064]]

Final stacked ensemble:
[[63767, 115], [220, 322038]]

Base models inside the stack:
RF: [[63740, 142], [301, 321957]]
HGB: [[63798, 84], [257, 322001]]
LR: [[40548, 23334], [17073, 305185]]
XGB: [[63774, 108], [203, 322055]]

With the matrix format [attack, benign], that means:

top-left = true attacks detected
top-right = attacks missed
bottom-left = benign falsely flagged as attack
bottom-right = benign correctly classified

So in plain words:

HGB had the fewest missed attacks: 84
RF was weaker than HGB and XGB
LR was much worse than the tree models
XGB inside the stack was very strong
Baseline XGBoost is still the best overall model to present as your main model
Final stacked ensemble slightly improved missed attacks compared to baseline XGBoost, but caused more false alarms, which is why its F1 ended up a bit worse overall

The short conclusion is:

The strongest individual models were XGBoost and HistGradientBoosting
Logistic Regression underperformed badly
The stacked ensemble did not outperform standalone XGBoost overall
Therefore, standalone XGBoost remains the chosen final model