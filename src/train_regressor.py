import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# Daten laden
train = pd.read_csv("data/regression_train.csv")
test  = pd.read_csv("data/regression_test.csv")

X_train, y_train = train.drop("preis", axis=1), train["preis"]
X_test,  y_test  = test.drop("preis", axis=1),  test["preis"]

# Modell trainieren
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluation
preds = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, preds):,.0f} EUR")
print(f"R²:   {r2_score(y_test, preds):.4f}")

model.save_model("xgboost_house_model.json")
print("Modell gespeichert: xgboost_house_model.json")
