"""California housing price prediction: full training pipeline.

Dataset: California housing (20640 districts, from the 1990 census).
Target: median house value (in $100,000s). Compares three regression
models on a held-out test split and with 3-fold cross-validation.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load
import os
if not os.path.exists("housing.csv"):
    print("housing.csv not found, generating from scikit-learn ...")
    from sklearn.datasets import fetch_california_housing
    fetch_california_housing(as_frame=True).frame.to_csv("housing.csv", index=False)

df = pd.read_csv("housing.csv")
print(f"Loaded {len(df)} districts, columns: {list(df.columns)}")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(3, shuffle=True, random_state=42)

# 2. Compare three models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
    results[name] = {
        "r2": r2_score(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "mae": mean_absolute_error(y_test, pred),
        "cv_r2": cv_r2,
    }
    fitted[name] = model
    r = results[name]
    print(f"{name:20s} R2={r['r2']:.3f} RMSE={r['rmse']:.3f} MAE={r['mae']:.3f} CV_R2={cv_r2:.3f}")

best = max(results, key=lambda k: results[k]["r2"])
print(f"\nBest model by R2: {best}  (values are in $100,000s)")

# 3. Plots: actual vs predicted + feature importances of the best model
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pred = fitted[best].predict(X_test)
axes[0].scatter(y_test, pred, alpha=0.2, s=8)
axes[0].plot([0, 5], [0, 5], "r--")
axes[0].set_xlabel("actual value ($100k)")
axes[0].set_ylabel("predicted value ($100k)")
axes[0].set_title(f"{best} - actual vs predicted")

imp = pd.Series(fitted[best].feature_importances_, index=X.columns).sort_values()
imp.plot.barh(ax=axes[1])
axes[1].set_title(f"{best} - feature importances")
plt.tight_layout()
plt.savefig("results.svg")
print("Saved results.svg")
