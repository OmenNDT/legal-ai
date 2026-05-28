# OUTLINE BÁO CÁO ĐỒ ÁN MÁY HỌC

> **Đề tài**: Dự đoán độ trễ giao hàng (Trip Delay Prediction) trong vận tải đường bộ
> **Sinh viên**: Trần Tín Nghĩa - UIT
> **Môn**: Máy học

---

## TRANG BÌA

- Tên đề tài
- Họ tên SV, MSSV, Lớp
- GV hướng dẫn
- Khoa, Trường
- Năm học

## MỤC LỤC

## DANH MỤC HÌNH ẢNH

## DANH MỤC BẢNG

---

# CHƯƠNG 1: TÓM TẮT ĐỀ TÀI (1 trang)

- Tóm tắt mục tiêu đồ án
- Phương pháp sử dụng (Random Forest, XGBoost, LightGBM)
- Kết quả chính đạt được
- Ý nghĩa thực tiễn

# CHƯƠNG 2: GIỚI THIỆU (3-4 trang)

## 2.1. Đặt vấn đề
- Tầm quan trọng của ngành logistics
- Vấn đề delay trong vận tải đường bộ
- Chi phí ước tính do delay (US logistics: ~$74 billion/năm do delay)

## 2.2. Đề tài
**Tên đề tài**: Dự đoán số giờ trễ của chuyến hàng (Trip Delay Hours Prediction)

### Bài toán cụ thể
- Đầu vào: Thông tin về chuyến hàng (load, trip, driver, truck, route, customer, facility, weather pattern...)
- Đầu ra: Dự đoán số giờ delay (regression)
- Loại: Supervised learning - Regression

## 2.3. Khó khăn và thách thức

### Thách thức về dữ liệu
1. **Đa nguồn (14 bảng)**: Phải join nhiều bảng, xử lý quan hệ phức tạp
2. **Dữ liệu lớn**: 361K+ records, ~57K trips → cần xử lý hiệu quả
3. **Mixed data types**: Numerical, categorical, datetime, text
4. **Missing values**: 2% null trong driver/truck assignments
5. **Outliers**: Trip delay extreme do incidents, breakdowns

### Thách thức về mô hình
1. **Non-linear relationships**: Delay phụ thuộc vào tương tác phức tạp giữa nhiều yếu tố
2. **Temporal patterns**: Seasonal, day-of-week, hour-of-day
3. **High cardinality categorical**: 150 drivers, 120 trucks, 200 customers
4. **Feature engineering nặng**: Cần tạo lag features, aggregate features

### Thách thức về đánh giá
1. **Time-based validation**: Phải tránh data leakage
2. **Skewed target distribution**: Hầu hết trips không delay nhiều
3. **Business interpretation**: Cần giải thích được model

## 2.4. Ý nghĩa thực tiễn

### Cho doanh nghiệp logistics
1. **Cảnh báo sớm**: Phát hiện trip có nguy cơ delay → reroute
2. **SLA management**: Đảm bảo cam kết khách hàng
3. **Resource optimization**: Match driver/truck phù hợp
4. **Pricing accuracy**: Quote giá phản ánh đúng risk
5. **Customer satisfaction**: Giảm delay → tăng retention

### Cho khách hàng (shippers)
1. **Visibility**: Biết trước khả năng delay để lên kế hoạch
2. **Inventory planning**: Tránh stockout
3. **Cost saving**: Giảm chi phí buffer

### Lợi ích kinh tế
- Mỗi 1% giảm delay → tiết kiệm ~$740M cho ngành Mỹ
- ROI rõ ràng và đo lường được

## 2.5. Phạm vi đồ án

- Dataset: Logistics Operations Database 2022-2024
- Bài toán: Regression (dự đoán giờ delay)
- 3 mô hình: Random Forest, XGBoost, LightGBM
- Đánh giá: MAE, RMSE, R², MAPE

---

# CHƯƠNG 3: CÁC PHƯƠNG PHÁP TIẾP CẬN (3-4 trang)

## 3.1. Tổng quan các phương pháp giải quyết bài toán delay prediction

### 3.1.1. Phương pháp truyền thống (Statistical/Time-series)
- **ARIMA, SARIMA**: Time-series classical
  - Ưu: Đơn giản, giải thích được
  - Nhược: Chỉ dựa trên lịch sử target, không tận dụng features
- **Linear Regression, Ridge, Lasso**:
  - Ưu: Nhanh, baseline tốt, giải thích được
  - Nhược: Không capture non-linearity, kém với dữ liệu phức tạp

### 3.1.2. Phương pháp Machine Learning cổ điển
- **Decision Tree**: Đơn giản nhưng overfit
- **Random Forest**: Bagging, robust, ít overfit
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost - SOTA cho tabular
- **SVR (Support Vector Regression)**: Mạnh nhưng chậm với data lớn
- **KNN Regression**: Đơn giản, kém scale

### 3.1.3. Deep Learning
- **MLP (Multi-Layer Perceptron)**: Generic neural network
- **LSTM/GRU**: Cho sequential data, thời gian
- **TabNet, FT-Transformer**: SOTA deep learning cho tabular
- Ưu: Capture rất phức tạp
- Nhược: Cần data lớn, khó interpret, kém hơn tree-based trên tabular

### 3.1.4. Hybrid approaches
- Stacking ensemble
- Blending multiple models
- Meta-learning

## 3.2. Phương pháp nhóm chọn

Nhóm chọn **3 mô hình tree-based**:
1. **Random Forest Regressor** - Bagging baseline
2. **XGBoost Regressor** - Champion gradient boosting
3. **LightGBM Regressor** - Fast challenger

### 3.2.1. Lý do tổng quát chọn tree-based
- ✅ SOTA cho tabular data (Kaggle competitions chứng minh)
- ✅ Xử lý mixed data types tự nhiên
- ✅ Capture non-linear interactions
- ✅ Robust với outliers
- ✅ Feature importance dễ giải thích
- ✅ Không cần feature scaling phức tạp
- ✅ Train nhanh, deploy được

### 3.2.2. Lý do chọn 3 mô hình cụ thể

#### Random Forest
- Đại diện cho **Bagging ensemble**
- Baseline mạnh, ít cần tuning
- So sánh với boosting để thấy ưu điểm boosting
- Variance reduction tốt

#### XGBoost
- Đại diện cho **Gradient Boosting (level-wise)**
- Thuật toán thắng nhiều competitions
- Built-in regularization
- SHAP support tốt

#### LightGBM
- Đại diện cho **Gradient Boosting (leaf-wise)**
- Khác biệt thuật toán so với XGBoost
- Native categorical features
- Tốc độ và memory efficiency

### 3.2.3. So sánh 3 model

| Tiêu chí | Random Forest | XGBoost | LightGBM |
|----------|---------------|---------|----------|
| Loại | Bagging | Boosting (level-wise) | Boosting (leaf-wise) |
| Tốc độ | Trung bình | Nhanh | Nhanh nhất |
| Accuracy | Tốt | Rất tốt | Rất tốt |
| Categorical | Cần encode | Cần encode | Native |
| Memory | Cao | Trung bình | Thấp |
| Interpretability | Tốt | Tốt (SHAP) | Tốt (SHAP) |

---

# CHƯƠNG 4: THỰC NGHIỆM (10-12 trang)

## 4.1. Dataset

### 4.1.1. Mô tả tổng quan
- Tên: Logistics Operations Database (2022-2024)
- Nguồn: Kaggle (Yogape Rodriguez)
- Tổng records: 361,799
- Số bảng: 14 bảng (relational database)
- Thời gian: 2022-2024

### 4.1.2. Schema 14 bảng
*[Bảng chi tiết các bảng và quan hệ]*

| Bảng | Records | Vai trò |
|------|---------|---------|
| trips | 57,096 | Core fact table - chứa target |
| loads | 57,096 | Shipment info |
| ... | ... | ... |

### 4.1.3. Số mẫu
- **Tổng samples cho bài toán**: 57,096 trips
- **Sau khi loại nulls/invalid**: ước tính ~55,000 trips
- **Phân chia**:
  - Train: 70% (~38,500)
  - Validation: 15% (~8,250)
  - Test: 15% (~8,250)

### 4.1.4. Phân tích target
- Target: `delay_hours` (continuous, có thể âm/dương)
- Distribution: Skewed right (long tail)
- Outliers: Có (do incidents, breakdowns)

### 4.1.5. Phân chia train/val/test
- **Strategy**: Time-based split (chronological)
  - Train: 2022 + nửa đầu 2023
  - Val: Nửa cuối 2023
  - Test: 2024
- **Lý do**: Tránh data leakage, mô phỏng deployment thực tế

## 4.2. EDA (Exploratory Data Analysis)

### 4.2.1. Univariate analysis
- Distribution của target (delay_hours)
- Distribution của các features chính
- Histogram, boxplot, density plots

### 4.2.2. Bivariate analysis
- Correlation matrix
- Scatter plots với target
- Categorical vs target (boxplots)

### 4.2.3. Temporal analysis
- Delay theo tháng/quý
- Delay theo ngày trong tuần
- Delay theo giờ
- Seasonal patterns

### 4.2.4. Geographic analysis
- Delay theo route
- Delay theo origin/destination

### 4.2.5. Driver/Truck analysis
- Top drivers có delay cao/thấp
- Truck age vs delay
- Maintenance frequency vs delay

### 4.2.6. Outlier analysis
- IQR method
- Z-score
- Isolation Forest

## 4.3. Tiền xử lý dữ liệu

### 4.3.1. Data cleaning
- Xử lý missing values
  - Drivers/trucks: 2% null → impute hoặc drop
  - Numerical: median imputation
  - Categorical: mode hoặc 'Unknown'
- Xử lý duplicates
- Xử lý invalid values (negative distance, etc.)

### 4.3.2. Outlier handling
- IQR capping cho numerical
- Log transformation cho skewed features
- Winsorization

### 4.3.3. Feature Engineering (Quan trọng nhất)

#### Driver features (từ drivers + driver_monthly_metrics)
- `driver_tenure_days`: Số ngày làm việc
- `driver_avg_mpg_3m`: MPG trung bình 3 tháng
- `driver_on_time_rate_3m`: Tỷ lệ on-time 3 tháng
- `driver_total_trips`: Tổng số trips
- `driver_incident_count`: Số incidents

#### Truck features (từ trucks + maintenance + truck_metrics)
- `truck_age_years`
- `truck_total_miles`
- `truck_maint_count_90d`: Số lần maintenance 90 ngày qua
- `truck_days_since_maint`: Ngày từ lần bảo trì cuối
- `truck_utilization_3m`: Utilization 3 tháng

#### Route features (từ routes + historical trips)
- `route_distance`
- `route_avg_delay_hist`: Delay trung bình lịch sử
- `route_traffic_complexity`: Complexity score

#### Temporal features
- `month`, `quarter`, `day_of_week`, `hour_of_day`
- `is_weekend`, `is_holiday`
- `season`: Spring, Summer, Fall, Winter

#### Customer features
- `customer_avg_detention`: Detention trung bình
- `customer_payment_terms`
- `customer_revenue_potential`

#### Facility features
- `origin_dock_doors`, `dest_dock_doors`
- `facility_type` (terminal/warehouse)

#### Lag features
- `prev_trip_delay_driver`
- `prev_trip_delay_truck`
- `prev_trip_delay_route`

### 4.3.4. Encoding
- **Label Encoding**: cho ordinal categorical
- **One-Hot Encoding**: cho low cardinality
- **Target Encoding**: cho high cardinality (driver_id, truck_id)
- **LightGBM native**: Sử dụng cho LightGBM

### 4.3.5. Feature scaling
- StandardScaler cho Linear models (nếu thử baseline)
- Tree-based models KHÔNG cần scaling

### 4.3.6. Feature selection
- Correlation threshold
- Feature importance từ baseline model
- Recursive Feature Elimination (RFE)

## 4.4. Phương pháp đánh giá

### 4.4.1. Metrics chính
- **MAE** (Mean Absolute Error)
  - Công thức: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
  - Ý nghĩa: Sai số trung bình tuyệt đối (giờ)
- **RMSE** (Root Mean Squared Error)
  - Công thức: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
  - Ý nghĩa: Penalize outliers
- **R²** (R-squared)
  - Công thức: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
  - Ý nghĩa: Phương sai giải thích được
- **MAPE** (Mean Absolute Percentage Error)
  - Công thức: $MAPE = \frac{100\%}{n}\sum_{i=1}^{n}|\frac{y_i - \hat{y}_i}{y_i}|$

### 4.4.2. Validation strategy
- **Time-based 5-fold CV**: TimeSeriesSplit
- **Holdout test set**: Final evaluation

### 4.4.3. Hyperparameter tuning
- **Method**: Optuna (Bayesian optimization)
- **Trials**: 50-100 trials per model
- **Objective**: Minimize MAE on validation

### 4.4.4. Phân tích bổ sung
- Residual plots
- Feature importance (SHAP)
- Error analysis by segment
- Train vs Validation curves

## 4.5. Setup thực nghiệm

### 4.5.1. Environment
- Python 3.10+
- 16GB RAM
- CPU/GPU
- Libraries: pandas, scikit-learn, xgboost, lightgbm, optuna, shap

### 4.5.2. Reproducibility
- Random seed: 42
- Save preprocessed data
- Save trained models
- Log all metrics

## 4.6. Kết quả thực nghiệm

### 4.6.1. Kết quả Random Forest
- Best params
- Train/Val/Test metrics
- Feature importance
- Residual plot
- Error analysis

### 4.6.2. Kết quả XGBoost
- Best params
- Train/Val/Test metrics
- SHAP analysis
- Residual plot

### 4.6.3. Kết quả LightGBM
- Best params
- Train/Val/Test metrics
- Feature importance
- Residual plot

### 4.6.4. So sánh 3 mô hình

| Model | Train MAE | Val MAE | Test MAE | Test RMSE | Test R² | Test MAPE | Train Time |
|-------|-----------|---------|----------|-----------|---------|-----------|------------|
| Random Forest | ? | ? | ? | ? | ? | ? | ? |
| XGBoost | ? | ? | ? | ? | ? | ? | ? |
| LightGBM | ? | ? | ? | ? | ? | ? | ? |

### 4.6.5. Phân tích kết quả

**Câu hỏi cần trả lời:**
1. Model nào tốt nhất? Vì sao?
2. Chênh lệch giữa các model là bao nhiêu?
3. Phân tích metrics chi tiết
4. Phân tích residuals
5. Có overfitting/underfitting không?
6. Top features ảnh hưởng nhất là gì?
7. Model hay sai trong segment nào?
8. Yếu tố gì ảnh hưởng đến kết quả?

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (2 trang)

## 5.1. Kết luận

### 5.1.1. Tóm tắt kết quả
- Đã xây dựng thành công pipeline ML cho delay prediction
- Đạt được MAE = X giờ trên test set
- LightGBM/XGBoost vượt trội Random Forest

### 5.1.2. Đóng góp của đồ án
- Tận dụng được toàn bộ 14 bảng dữ liệu
- Pipeline feature engineering chi tiết
- So sánh có hệ thống 3 mô hình tree-based
- Insights về delay patterns

### 5.1.3. Hạn chế
- Chưa có dữ liệu thời tiết
- Chưa có dữ liệu traffic real-time
- Dataset synthetic (không phải real production)
- Chưa thử deep learning

## 5.2. Hướng phát triển

### 5.2.1. Cải thiện model
- Stacking ensemble (RF + XGB + LGB)
- Neural networks (TabNet, FT-Transformer)
- Deep learning với LSTM cho time-series
- AutoML (H2O, AutoGluon)

### 5.2.2. Mở rộng features
- Tích hợp weather data (NOAA API)
- Traffic data (Google Maps API)
- Holiday calendar
- Economic indicators (fuel price, GDP)

### 5.2.3. Cải thiện pipeline
- MLOps: MLflow, DVC
- Real-time serving (FastAPI)
- Monitoring & retraining
- A/B testing

### 5.2.4. Mở rộng bài toán
- Multi-task: Delay + Cost + Profit
- Anomaly detection cho safety incidents
- Driver retention prediction
- Route optimization

---

# CHƯƠNG 6: TÀI LIỆU THAM KHẢO

[Xem REFERENCES/references.md]

---

# PHỤ LỤC

## A. Mã nguồn
*[Trỏ tới SOURCE_CODE/]*

## B. Notebook outputs
## C. Visualizations bổ sung
## D. Hyperparameters tốt nhất
