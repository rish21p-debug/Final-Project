# AI System: Heart Disease Classification
# Dataset: Synthetic heart disease-like tabular data
# Model: Random Forest Classifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Generate a synthetic heart disease-like dataset."""
    np.random.seed(42)
    n = 500

    age = np.random.randint(29, 77, n)
    sex = np.random.randint(0, 2, n)
    cp = np.random.randint(0, 4, n)
    trestbps = np.random.randint(94, 200, n)
    chol = np.random.randint(126, 564, n)
    fbs = np.random.randint(0, 2, n)
    restecg = np.random.randint(0, 3, n)
    thalach = np.random.randint(71, 202, n)
    exang = np.random.randint(0, 2, n)
    oldpeak = np.round(np.random.uniform(0, 6.2, n), 1)
    slope = np.random.randint(0, 3, n)
    ca = np.random.randint(0, 4, n)
    thal = np.random.randint(0, 3, n)

    X = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    })

    # Simulate disease label with some logic
    score = (
        (age > 55).astype(int) +
        (sex == 1).astype(int) +
        (cp > 1).astype(int) * 2 +
        (thalach < 140).astype(int) +
        (exang == 1).astype(int) +
        (ca > 0).astype(int) * 2 +
        (oldpeak > 2).astype(int)
    )
    y = pd.Series((score >= 4).astype(int), name='target')

    return X, y

def preprocess(X_train, X_test):
    """Scale features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc

def main():
    print("=== Heart Disease Classification System ===\n")

    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}, Class distribution: {y.value_counts().to_dict()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Preprocessing...")
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    print("Training model...")
    model = train_model(X_train_s, y_train)

    print("Evaluating model...")
    acc = evaluate_model(model, X_test_s, y_test)

    print(f"\n✓ System running successfully. Final accuracy: {acc:.4f}")
    return model, scaler

if __name__ == "__main__":
    main()

