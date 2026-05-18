import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Daten laden
train = pd.read_csv("data/classification_train.csv")
test  = pd.read_csv("data/classification_test.csv")

X_train, y_train = train.drop("churn", axis=1), train["churn"]
X_test,  y_test  = test.drop("churn", axis=1),  test["churn"]

# Modell trainieren
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluation
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=["Kein Churn", "Churn"]))

model.save_model("xgboost_churn_model.json")
print("Modell gespeichert: xgboost_churn_model.json")
