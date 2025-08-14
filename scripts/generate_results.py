import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

# Step 1: Import optimizers
from scripts.pso_optimizer import pso_optimizer
from scripts.aco_optimizer import aco_optimizer
from scripts.hybrid_optimizer import run_hybrid_optimizer

# Step 2: Load dataset
df = pd.read_csv('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/data/data.csv')

# Step 3: Clean and preprocess
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Step 4: Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Define fitness function for hybrid optimizer
def fitness_function(c_val):
    model = LogisticRegression(C=c_val, max_iter=1000, solver='liblinear')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Step 6: Run optimizers
print("Running PSO...")
pso_result = pso_optimizer(X_train, y_train, X_test, y_test)

print("Running ACO...")
aco_result = aco_optimizer(X_train, y_train, X_test, y_test)

print("Running Hybrid (ACO + GridSearch)...")
hybrid_result = run_hybrid_optimizer(X_train, y_train, X_test, y_test)

# Step 7: Evaluate results
results = []
conf_matrices = {}
exec_times = {}

for name, result in {
    "PSO": pso_result,
    "ACO": aco_result,
    "Hybrid": hybrid_result
}.items():
    y_pred = result["model"].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Optimizer": name,
        "Best_C": round(result["best_c"], 4),
        "Accuracy": round(acc, 4)
    })
    conf_matrices[name] = cm
    exec_times[name] = round(result["execution_time"], 4)

# Step 8: Save results
os.makedirs('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/results', exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/results/model_performance.csv', index=False)

# Step 9: Plot confusion matrix and Accuracy vs Execution Time
os.makedirs('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/results/images', exist_ok=True)

# Save confusion matrix plots
for name, cm in conf_matrices.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/results/images/{name}_confusion_matrix.png')
    plt.close()

# Save Plot Accuracy vs Execution Time
os.makedirs('result/images', exist_ok=True)

optimizers = [r["Optimizer"] for r in results]
accuracies = [r["Accuracy"] for r in results]
times = [exec_times[r["Optimizer"]] for r in results]

fig, ax1 = plt.subplots(figsize=(7, 4))

ax2 = ax1.twinx()
ax1.bar(optimizers, accuracies, color='skyblue', width=0.4, align='center', label='Accuracy')
ax2.plot(optimizers, times, color='red', marker='o', linewidth=2, label='Execution Time (s)')

ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Execution Time (s)')
ax1.set_ylim(0.9, 1.01)
ax1.set_title('Optimizer Accuracy vs Execution Time')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

plt.tight_layout()
plt.savefig('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/results/images/accuracy_vs_time.png')
plt.close()

print("All results saved in the 'results/result' folder including accuracy vs time plot.")
