# KẾ HOẠCH ĐỒ ÁN MÁY HỌC

## 📋 Thông tin đồ án

- **Môn học**: Máy học (UIT)
- **Đề tài**: Dự đoán độ trễ giao hàng (Trip Delay Hours Prediction) trong vận tải đường bộ
- **Bài toán**: Regression
- **Dataset**: Logistics Operations Database (2022-2024) - Kaggle
- **Sinh viên**: Trần Tín Nghĩa

---

## 🎯 1. Mục tiêu đồ án

1. Hiểu và phân tích bộ dữ liệu Logistics Operations (14 bảng, 361K+ records)
2. Thực hiện EDA, tiền xử lý, feature engineering từ 14 bảng
3. Áp dụng 3 mô hình ML: **Random Forest**, **XGBoost**, **LightGBM**
4. So sánh hiệu suất và chọn mô hình tốt nhất
5. Phân tích kết quả và đưa ra insights kinh doanh

---

## 📊 2. Bài toán & Dataset

### 2.1. Bài toán
**Trip Delay Hours Prediction** - Dự đoán số giờ delay của một chuyến hàng (trip) so với thời gian dự kiến.

- **Loại bài toán**: Regression (supervised learning)
- **Target**: `delay_hours = actual_arrival - scheduled_arrival` (đơn vị: hours)
- **Input**: 14 bảng dữ liệu logistics joined thành master dataset
- **Output**: Dự đoán delay (số dương = trễ, âm = sớm, 0 = đúng giờ)

### 2.2. Dataset

- **Nguồn**: Kaggle - Logistics Operations Database (2022-2024)
- **Tổng records**: 361,799
- **Tổng bảng**: 14 (drivers, trucks, trailers, customers, facilities, routes, loads, trips, fuel_purchases, maintenance_records, delivery_events, safety_incidents, driver_monthly_metrics, truck_utilization_metrics)
- **Sample chính**: 57,096 trips
- **Thời gian**: 2022-2024 (3 năm)
- **Dùng được toàn bộ 14/14 bảng** thông qua feature engineering

### 2.3. Phân chia dữ liệu

- **Train**: 70% (~40,000 trips, 2022-2023 đầu)
- **Validation**: 15% (~8,500 trips, 2023 cuối)
- **Test**: 15% (~8,500 trips, 2024)
- **Phương pháp**: Time-based split (tránh data leakage)

---

## 🤖 3. Lựa chọn 3 Mô hình ML

### 3.1. Random Forest Regressor (Baseline)

**Lý do chọn:**
- Bagging ensemble - đối chứng với boosting
- Robust, ít overfitting, ít tuning
- Feature importance dễ giải thích
- Hiệu năng tốt trên tabular data

**Hyperparameters tune:**
- `n_estimators`: [100, 300, 500]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `max_features`: ['sqrt', 'log2']

### 3.2. XGBoost Regressor (Champion)

**Lý do chọn:**
- State-of-the-art cho tabular data
- Xử lý non-linear interactions tốt
- Built-in regularization (L1, L2)
- Hỗ trợ SHAP cho interpretability
- Robust với outliers

**Hyperparameters tune:**
- `n_estimators`: [200, 500, 1000]
- `max_depth`: [4, 6, 8]
- `learning_rate`: [0.01, 0.05, 0.1]
- `subsample`: [0.7, 0.8, 0.9]
- `colsample_bytree`: [0.7, 0.8, 0.9]

### 3.3. LightGBM Regressor (Challenger)

**Lý do chọn:**
- Leaf-wise tree growth (khác level-wise của XGBoost)
- Native categorical support
- Memory efficient, train nhanh nhất
- Hiệu năng tương đương XGBoost
- Phù hợp với high-cardinality features (driver_id, truck_id)

**Hyperparameters tune:**
- `n_estimators`: [200, 500, 1000]
- `num_leaves`: [31, 63, 127]
- `learning_rate`: [0.01, 0.05, 0.1]
- `feature_fraction`: [0.7, 0.8, 0.9]
- `bagging_fraction`: [0.7, 0.8, 0.9]

---

## 📈 4. Phương pháp đánh giá

### 4.1. Metrics chính (Regression)

- **MAE** (Mean Absolute Error): Sai số trung bình tuyệt đối (giờ)
- **RMSE** (Root Mean Squared Error): Penalize outliers
- **R²** (Coefficient of Determination): Phương sai giải thích được
- **MAPE** (Mean Absolute Percentage Error): Sai số tương đối

### 4.2. Validation strategy

- **Time-based K-Fold** (k=5): Tôn trọng tính time-series
- **Holdout test set**: 2024 data (chưa thấy)

### 4.3. Phân tích bổ sung

- **Residual analysis**: Phân bố sai số
- **Feature importance**: Top features theo SHAP/Gain
- **Error analysis**: Trips delay nhiều/ít, theo route/driver/customer
- **Train vs Val**: Phát hiện overfitting/underfitting

---

## 🗂️ 5. Cấu trúc Project

```
DoAnMonHoc/
├── PLAN.md                        # File này
├── README.md                      # Hướng dẫn sử dụng
│
├── DOCS/                          # 📄 Báo cáo Word
│   ├── BaoCao_DoAn.docx           # Báo cáo chính
│   ├── outline.md                 # Outline 10 mục
│   └── figures/                   # Hình ảnh dùng trong báo cáo
│
├── SLIDES/                        # 🎤 Slide thuyết trình
│   ├── Slides_DoAn.pptx
│   └── outline.md
│
├── SOURCE_CODE/                   # 💻 Code
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Preprocessing.ipynb
│   │   ├── 03_Feature_Engineering.ipynb
│   │   ├── 04_Modeling_RandomForest.ipynb
│   │   ├── 05_Modeling_XGBoost.ipynb
│   │   ├── 06_Modeling_LightGBM.ipynb
│   │   └── 07_Comparison_Analysis.ipynb
│   ├── src/                       # Python modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── models.py
│   │   ├── evaluation.py
│   │   └── utils.py
│   ├── data/
│   │   ├── raw/                   # CSV gốc từ Kaggle
│   │   ├── processed/             # Sau preprocessing
│   │   └── features/              # Master dataset cho modeling
│   ├── models/                    # Saved models (.pkl)
│   ├── results/
│   │   ├── figures/               # Plots
│   │   └── metrics/               # JSON/CSV metrics
│   └── requirements.txt
│
└── REFERENCES/                    # 📚 Tài liệu tham khảo
    └── references.md
```

---

## 📅 6. Roadmap thực hiện

### Phase 1: Setup & Data Understanding (Tuần 1)
- [x] Tạo cấu trúc project
- [ ] Download dataset từ Kaggle
- [ ] Khám phá schema 14 bảng
- [ ] Hiểu relationships giữa các bảng

### Phase 2: EDA (Tuần 1-2)
- [ ] Notebook 01: EDA chi tiết
  - Distribution của target (delay_hours)
  - Phân tích từng bảng
  - Correlations
  - Time series patterns
  - Outliers detection
- [ ] Tạo 15-20 visualizations

### Phase 3: Preprocessing & Feature Engineering (Tuần 2-3)
- [ ] Notebook 02: Data cleaning
  - Missing values
  - Outliers
  - Data types
- [ ] Notebook 03: Feature Engineering
  - Join 14 bảng → master dataset
  - Driver features (tenure, on-time history)
  - Truck features (age, maintenance freq)
  - Route features (historical avg)
  - Temporal features (month, dow, season)
  - Customer features
  - Facility features
  - Lag features

### Phase 4: Modeling (Tuần 3-4)
- [ ] Notebook 04: Random Forest
  - Train baseline
  - Hyperparameter tuning (GridSearch/Optuna)
  - Save model
- [ ] Notebook 05: XGBoost
  - Train + tune
  - Early stopping
  - SHAP analysis
- [ ] Notebook 06: LightGBM
  - Train + tune
  - Categorical native handling

### Phase 5: Analysis & Comparison (Tuần 4)
- [ ] Notebook 07: So sánh 3 models
  - Bảng metrics
  - Residual plots
  - Feature importance comparison
  - Error analysis
  - Conclusion

### Phase 6: Documentation (Tuần 5)
- [ ] Viết báo cáo Word (10 mục)
- [ ] Tạo slides
- [ ] Hoàn thiện code
- [ ] Cleanup repo

---

## 📝 7. Outline Báo cáo Word (10 mục)

1. **Tóm tắt đề tài** (1 trang)
2. **Giới thiệu bài toán & dataset** (2-3 trang)
   - Bài toán là gì
   - Khó khăn & thách thức
   - Ý nghĩa thực tiễn
   - Mô tả dataset
3. **Các phương pháp tiếp cận** (2 trang)
   - Linear models
   - Tree-based
   - Neural networks
   - Phương pháp nhóm chọn + lý do
4. **EDA + Biểu đồ** (4-5 trang)
5. **Tiền xử lý dữ liệu** (3 trang)
6. **Mô hình & Thông số** (3-4 trang)
7. **Kết quả & Phân tích** (4-5 trang)
8. **So sánh mô hình** (2-3 trang)
9. **Kết luận & Hướng phát triển** (2 trang)
10. **Tài liệu tham khảo + Phụ lục code** (2 trang)

**Tổng: ~25-30 trang**

---

## 🎤 8. Outline Slide (15-20 slides)

1. Title slide
2. Tổng quan đề tài
3. Bài toán & ý nghĩa thực tiễn
4. Dataset overview
5. EDA highlights (3-4 slides)
6. Pipeline tổng thể
7. Feature engineering
8. 3 mô hình & lý do chọn
9. Kết quả Random Forest
10. Kết quả XGBoost
11. Kết quả LightGBM
12. So sánh tổng hợp
13. Feature importance
14. Error analysis
15. Kết luận
16. Hướng phát triển
17. Q&A

---

## ✅ 9. Sản phẩm nộp

- [ ] **DOCS/BaoCao_DoAn.docx** - Báo cáo Word đầy đủ 10 mục
- [ ] **SLIDES/Slides_DoAn.pptx** - Slide thuyết trình
- [ ] **SOURCE_CODE/** - Toàn bộ code
  - 7 Jupyter notebooks
  - Python modules
  - requirements.txt
  - README.md
- [ ] **README.md** - Hướng dẫn cài đặt & chạy

---

## 🛠️ 10. Tech Stack

- **Python**: 3.10+
- **Data**: pandas, numpy, polars (cho dataset lớn)
- **Visualization**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, xgboost, lightgbm
- **Tuning**: optuna, scikit-optimize
- **Interpretability**: shap, eli5
- **Notebook**: jupyter, jupyterlab
- **Document**: python-docx (auto-gen), python-pptx
