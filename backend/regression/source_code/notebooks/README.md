# Notebooks - Hướng dẫn chạy theo thứ tự

## Thứ tự thực hiện

1. **01_EDA.ipynb** - Khám phá dữ liệu
   - Load 14 bảng
   - Phân tích từng bảng
   - Phân tích target
   - Visualizations
   - Insights

2. **02_Preprocessing.ipynb** - Tiền xử lý
   - Missing values
   - Outliers
   - Data types
   - Time-based split

3. **03_Feature_Engineering.ipynb** - Feature Engineering
   - Join 14 bảng
   - Driver features
   - Truck features
   - Route features
   - Temporal features
   - Lag features
   - Save master dataset

4. **04_Modeling_RandomForest.ipynb**
   - Train baseline RF
   - Hyperparameter tuning
   - Evaluation
   - Save model

5. **05_Modeling_XGBoost.ipynb**
   - Train XGBoost
   - Hyperparameter tuning với Optuna
   - SHAP analysis
   - Save model

6. **06_Modeling_LightGBM.ipynb**
   - Train LightGBM
   - Hyperparameter tuning
   - Native categorical
   - Save model

7. **07_Comparison_Analysis.ipynb**
   - So sánh 3 models
   - Bảng metrics
   - Residual analysis
   - Feature importance comparison
   - Error analysis
   - Conclusion
