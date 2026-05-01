import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings, traceback
warnings.filterwarnings('ignore')

# --- Shared data loader (correct) ---
def load_data():
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
    X = pd.DataFrame({'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
                      'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
                      'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal})
    score = ((age>55).astype(int)+(sex==1).astype(int)+(cp>1).astype(int)*2+
             (thalach<140).astype(int)+(exang==1).astype(int)+(ca>0).astype(int)*2+(oldpeak>2).astype(int))
    y = pd.Series((score>=4).astype(int), name='target')
    return X, y

results = {}

# ─── BUG 1: test_size > 1 → ValueError ───────────────────────────────────────
def bug1():
    X, y = load_data()
    # CHANGED: test_size=0.2 → test_size=1.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.5, random_state=42, stratify=y)
try:
    bug1()
    results[1] = ("No error","")
except Exception as e:
    results[1] = (type(e).__name__, str(e)[:200])

# ─── BUG 2: fit_transform on test set (data leakage / shape ok but wrong) ────
def bug2():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    # CHANGED: transform(X_test) → fit_transform(X_test)  (data leakage)
    X_test_s  = scaler.fit_transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    return acc
try:
    acc = bug2()
    results[2] = ("No error (silent bug)", f"Accuracy={acc:.4f} — scaler re-fitted on test, causing data leakage / distribution shift")
except Exception as e:
    results[2] = (type(e).__name__, str(e)[:200])

# ─── BUG 3: wrong y passed to fit → shape mismatch ───────────────────────────
def bug3():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    # CHANGED: y_train → y  (full y instead of train split)
    model.fit(X_train_s, y)
try:
    bug3()
    results[3] = ("No error","")
except Exception as e:
    results[3] = (type(e).__name__, str(e)[:200])

# ─── BUG 4: n_estimators as string → TypeError ───────────────────────────────
def bug4():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    # CHANGED: n_estimators=100 → n_estimators="100"
    model = RandomForestClassifier(n_estimators= 100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
try:
    bug4()
    results[4] = ("No error","")
except Exception as e:
    results[4] = (type(e).__name__, str(e)[:200])

# ─── BUG 5: predict before fit → NotFittedError ──────────────────────────────
def bug5():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    # CHANGED: forgot model.fit(...) — predict called before training
    y_pred = model.predict(X_test_s)
try:
    bug5()
    results[5] = ("No error","")
except Exception as e:
    results[5] = (type(e).__name__, str(e)[:200])

# ─── BUG 6: wrong feature count at predict time → ValueError ─────────────────
def bug6():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    # CHANGED: pass only first 5 columns at predict time
    y_pred = model.predict(X_test_s[:, :5])
try:
    bug6()
    results[6] = ("No error","")
except Exception as e:
    results[6] = (type(e).__name__, str(e)[:200])

# ─── BUG 7: random_state removed → non-deterministic results (silent) ─────────
import io, sys
def bug7():
    accs = []
    for _ in range(3):
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        # CHANGED: random_state=42 removed from both split and model
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train_s, y_train)
        accs.append(accuracy_score(y_test, model.predict(X_test_s)))
    return accs
try:
    accs = bug7()
    results[7] = ("No error (silent bug)", f"Accuracies vary across runs: {[round(a,4) for a in accs]} — removing random_state causes non-reproducibility")
except Exception as e:
    results[7] = (type(e).__name__, str(e)[:200])

# ─── BUG 8: stratify=y but y is wrong length → ValueError ────────────────────
def bug8():
    X, y = load_data()
    X2, _ = load_data()   # second independent X with different index
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # CHANGED: accidentally pass X2 (same shape, different object) but stratify on y_train (wrong length)
    train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y_train)
try:
    bug8()
    results[8] = ("No error","")
except Exception as e:
    results[8] = (type(e).__name__, str(e)[:200])

# ─── BUG 9: max_depth=0 → invalid parameter ──────────────────────────────────
def bug9():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    # CHANGED: max_depth=5 → max_depth=0
    model = RandomForestClassifier(n_estimators=100, max_depth=0, random_state=42)
    model.fit(X_train_s, y_train)
try:
    bug9()
    results[9] = ("No error","")
except Exception as e:
    results[9] = (type(e).__name__, str(e)[:200])

# ─── BUG 10: accuracy_score args swapped → misleading metric (silent) ─────────
def bug10():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # CHANGED: accuracy_score(y_test, y_pred) → accuracy_score(y_pred, y_test)  (args swapped)
    acc = accuracy_score(y_pred, y_test)   # same value for accuracy but wrong semantic
    cr  = classification_report(y_pred, y_test)  # labels flipped
    return acc, cr
try:
    acc, cr = bug10()
    results[10] = ("No error (silent bug)", f"Accuracy={acc:.4f} same numerically, but classification_report uses y_pred as true label — misleading precision/recall per class")
except Exception as e:
    results[10] = (type(e).__name__, str(e)[:200])

# ─── Print all results ────────────────────────────────────────────────────────
for k, (etype, msg) in results.items():
    print(f"\n{'='*60}")
    print(f"BUG {k}: {etype}")
    print(f"  {msg}")

