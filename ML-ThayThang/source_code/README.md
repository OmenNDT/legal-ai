# Source Code - Trip Delay Prediction

## 📁 Cấu trúc

```
SOURCE_CODE/
├── notebooks/             # Jupyter notebooks chạy theo thứ tự
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling_RandomForest.ipynb
│   ├── 05_Modeling_XGBoost.ipynb
│   ├── 06_Modeling_LightGBM.ipynb
│   └── 07_Comparison_Analysis.ipynb
│
├── src/                   # Python modules tái sử dụng
│   ├── __init__.py
│   ├── data_loader.py     # Load 14 bảng CSV
│   ├── preprocessing.py   # Cleaning, missing, outliers, split
│   ├── feature_engineering.py  # Tạo features từ 14 bảng
│   ├── models.py          # 3 model definitions
│   ├── evaluation.py      # Metrics: MAE, RMSE, R2, MAPE
│   └── utils.py           # Helpers
│
├── data/
│   ├── raw/               # CSV gốc từ Kaggle
│   ├── processed/         # Sau preprocessing
│   └── features/          # Master dataset
│
├── models/                # Saved .pkl models
│
├── results/
│   ├── figures/           # Plots
│   └── metrics/           # JSON/CSV metrics
│
└── requirements.txt
```

## 🛠️ Setup

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=trip-delay
```

## 📥 Tải Dataset

```bash
# Cách 1: Kaggle CLI
cd data/raw
kaggle datasets download -d yogape/logistics-operations-database
unzip logistics-operations-database.zip

# Cách 2: Download thủ công từ
# https://www.kaggle.com/datasets/yogape/logistics-operations-database
# và giải nén vào data/raw/
```

Sau khi tải, `data/raw/` sẽ có:
```
drivers.csv, trucks.csv, trailers.csv, customers.csv,
facilities.csv, routes.csv, loads.csv, trips.csv,
fuel_purchases.csv, maintenance_records.csv,
delivery_events.csv, safety_incidents.csv,
driver_monthly_metrics.csv, truck_utilization_metrics.csv
```

## ▶️ Chạy notebooks

```bash
jupyter lab
```

Mở và chạy theo thứ tự `01_*` → `07_*`.

## 🔧 Sử dụng modules trực tiếp

```python
from src.data_loader import load_all_tables
from src.preprocessing import handle_missing_values, time_based_split
from src.feature_engineering import build_master_dataset
from src.models import get_random_forest, get_xgboost, get_lightgbm
from src.evaluation import regression_metrics, compare_models

# Load data
tables = load_all_tables("data/raw")

# Build features
df = build_master_dataset(tables)

# Train model
model = get_xgboost()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate
metrics = regression_metrics(y_test, preds)
print(metrics)
```
