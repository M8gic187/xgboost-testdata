# XGBoost Testdaten

Dieses Repository enthält Testdaten und Beispiele für XGBoost – aufgeteilt in Klassifikation und Regression.

## Struktur

```
├── data/
│   ├── classification_train.csv   # Binäre Klassifikation (Churn-Vorhersage)
│   ├── classification_test.csv
│   ├── regression_train.csv       # Regression (Hauspreise)
│   └── regression_test.csv
├── notebooks/
│   └── xgboost_example.ipynb      # Beispiel-Notebook
├── src/
│   ├── train_classifier.py        # XGBoost Klassifikation
│   └── train_regressor.py         # XGBoost Regression
└── requirements.txt
```

## Schnellstart

```bash
pip install -r requirements.txt
python src/train_classifier.py
python src/train_regressor.py
```

## Daten

### Klassifikation – Churn-Vorhersage
- **Features:** Alter, Vertragslaufzeit, monatliche Kosten, Supportanfragen, Produktanzahl
- **Ziel:** `churn` (0 = kein Abwandern, 1 = Abwandern)
- **Train:** 800 Zeilen | **Test:** 200 Zeilen

### Regression – Hauspreise
- **Features:** Zimmeranzahl, Wohnfläche, Baujahr, Entfernung Stadtzentrum, Garagenparkplätze
- **Ziel:** `preis` (in EUR)
- **Train:** 800 Zeilen | **Test:** 200 Zeilen
