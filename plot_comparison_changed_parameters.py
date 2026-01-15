import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

experiments = {
    "2 Qubits": "local_test_outputs/qdrl_uav_logs/exp_4u_2q",
    "4 Qubits": "local_test_outputs/qdrl_uav_logs/exp_4u_4q",
}

metric_filename = "secrecy_rates.csv"
# metric_filename = "ep_rewards.csv"       

results = {}
errors = {}

print("loading~~")

for label, folder in experiments.items():
    file_path = os.path.join(folder, metric_filename)
    
    if not os.path.exists(file_path):
        print(f"Warning: Can't find the file {file_path}")
        continue
        
    try:
        df = pd.read_csv(file_path)
        
        data = df.iloc[:, 0].values 
        last_n = int(len(data) * 0.2) # last 20%
        if last_n == 0: last_n = 1
        
        avg_score = np.mean(data[-last_n:]) # average
        std_dev = np.std(data[-last_n:])   
        
        results[label] = avg_score
        errors[label] = std_dev
        print(f"{label}: Average = {avg_score:.4f}")
        
    except Exception as e:
        print(f"Load {label} error: {e}")

# === plotting ===
labels = list(results.keys())
values = list(results.values())
stds = list(errors.values())

plt.figure(figsize=(8, 6))


bars = plt.bar(labels, values, yerr=stds, capsize=10, color=['#ff9999', '#66b3ff', '#99ff99'])

# 
plt.title("Impact of Qubit Count on Secrecy Rate (4 Users)", fontsize=14)
plt.ylabel("Average Secrecy Rate (Last 20% Episodes)", fontsize=12)
plt.xlabel("Quantum Model Configuration", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
