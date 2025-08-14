import pandas as pd
from scripts.hybrid_optimizer import run_hybrid_optimizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:/Users/ajayr/Desktop/Projects to upload/Metaheuristic_Optimization_for_Logistic_Regression/data/data.csv')

# Drop irrelevant columns
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Encode target
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

def test_hybrid_optimizer():
    best_params, accuracy = run_hybrid_optimizer(X_train, y_train, X_test, y_test)
    print(f"Hybrid Test | Best Params: {best_params}, Accuracy: {accuracy}")
    assert 0.0 < accuracy <= 1.0
