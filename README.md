# Real Estate Investment Advisor

Predicts whether a property is a **Good Investment** (classification) and its **price after 5 years** (regression), with a Streamlit app and MLflow tracking.

## Results (50,000-property test set)
| Task | Best model | Result |
| --- | --- | --- |
| Good Investment | XGBoost | Accuracy 82.3%, F1 0.786, ROC-AUC 0.925 |
| Price after 5 years | Random Forest | RMSE 1.28 lakh, MAE 0.84 lakh, R2 1.000 |

The targets are built from stated rules (the dataset has no future prices), so the regression score reflects the rule. See the documentation for details.

## Run
```bash
pip install -r requirements.txt
# put the merged dataset at data/india_housing_prices.csv (part1 + part2)
python 01_preprocessing.py --input data/india_housing_prices.csv --outdir outputs
python 02_eda.py --input outputs/cleaned_real_estate.csv --outdir outputs/eda
python 03_train_models.py --data outputs --outdir outputs/models
python 04_build_app_data.py
streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Merge the two dataset parts with:
```python
import pandas as pd
pd.concat([pd.read_csv("india_housing_prices_part1.csv"),
           pd.read_csv("india_housing_prices_part2.csv")]).to_csv("data/india_housing_prices.csv", index=False)
```

## Files
- `01_preprocessing.py` cleaning, features, targets, encoding, scaling
- `02_eda.py` the 20 EDA questions plus extras
- `03_train_models.py` training, evaluation, MLflow
- `04_build_app_data.py` small data files for the app
- `app.py`, `app_utils.py` Streamlit app
- `test_features.py`, `test_app.py` tests

## Deploy on Streamlit Community Cloud
Push to GitHub: all `.py` files, `requirements.txt`, `outputs/preprocessor.joblib`, `outputs/models/`, `outputs/app_data/`. Set the main file to `app.py`.
