# BÁO CÁO ĐỒ ÁN MÁY HỌC

## DỰ ĐOÁN ĐỘ TRỄ GIAO HÀNG TRONG VẬN TẢI ĐƯỜNG BỘ

### TRIP DELAY HOURS PREDICTION IN ROAD TRANSPORTATION

---

**Sinh viên thực hiện**: Trần Tín Nghĩa
**MSSV**: [Điền]
**Lớp**: [Điền]
**Giảng viên hướng dẫn**: [Điền]
**Khoa**: Khoa học Máy tính
**Trường**: Đại học Công nghệ Thông tin - ĐHQG TP.HCM
**Năm học**: 2025-2026

---

## TÓM TẮT

Đồ án này trình bày một giải pháp Machine Learning để dự đoán độ trễ giao hàng (trip delay) trong ngành vận tải đường bộ. Dựa trên bộ dữ liệu Logistics Operations Database (2022-2024) gồm 14 bảng và hơn 361,000 records, chúng tôi xây dựng pipeline xử lý dữ liệu và so sánh 3 mô hình tree-based: Random Forest, XGBoost và LightGBM. Mục tiêu là dự đoán số giờ delay của một chuyến hàng, qua đó hỗ trợ doanh nghiệp logistics tối ưu vận hành, đảm bảo SLA và tăng sự hài lòng khách hàng. Kết quả thực nghiệm cho thấy [model X] đạt hiệu suất tốt nhất với MAE = [X] giờ và R² = [X]. Đồ án minh họa quy trình ML hoàn chỉnh từ EDA, feature engineering đến đánh giá và phân tích lỗi.

**Từ khóa**: Trip Delay Prediction, Machine Learning, Random Forest, XGBoost, LightGBM, Logistics, Regression, Feature Engineering

---

## CHƯƠNG 1: GIỚI THIỆU

### 1.1. Đặt vấn đề

Ngành logistics và vận tải đường bộ đóng vai trò huyết mạch trong nền kinh tế hiện đại. Tại Mỹ, ngành trucking đóng góp hơn $940 tỷ mỗi năm và vận chuyển 72.5% tổng lượng hàng hóa. Tuy nhiên, **độ trễ giao hàng (delivery delay)** là một trong những vấn đề nhức nhối nhất, gây thiệt hại ước tính **hơn $74 tỷ mỗi năm** cho ngành vận tải Mỹ thông qua:

- Mất doanh thu do vi phạm SLA (Service Level Agreement)
- Chi phí phụ trội (detention, demurrage)
- Mất khách hàng do giảm satisfaction
- Hiệu ứng dây chuyền lên chuỗi cung ứng

Việc dự đoán chính xác delay sẽ giúp doanh nghiệp:
1. Cảnh báo sớm và can thiệp kịp thời
2. Lên kế hoạch resource hợp lý
3. Đưa ra quote giá phản ánh đúng risk
4. Cải thiện trải nghiệm khách hàng

### 1.2. Đề tài

**Tên đề tài**: Dự đoán độ trễ giao hàng (Trip Delay Hours Prediction) trong vận tải đường bộ sử dụng Machine Learning

**Mô tả bài toán**:
- **Input**: Thông tin về một chuyến hàng (load, trip, driver, truck, route, customer, facility, lịch sử...)
- **Output**: Dự đoán số giờ delay (số thực, có thể âm nếu sớm hơn dự kiến)
- **Loại bài toán**: Supervised Learning - Regression

### 1.3. Khó khăn và thách thức

#### 1.3.1. Thách thức về dữ liệu

**1. Dữ liệu đa nguồn từ 14 bảng**
- Phải join nhiều bảng theo các quan hệ phức tạp
- Cần hiểu rõ schema và business logic
- Risk of data leakage khi join sai cách

**2. Khối lượng dữ liệu lớn**
- 361,799 records, 57,096 trips
- 131K fuel purchases, 114K delivery events
- Cần xử lý hiệu quả về memory và compute

**3. Mixed data types**
- Numerical: revenue, distance, miles, costs
- Categorical: status, type, terminal
- Datetime: scheduled vs actual times
- Text: descriptions

**4. Missing values và outliers**
- 2% null trong assignments (intentional, mô phỏng thực tế)
- Outliers trong delay (do incidents, breakdowns)

#### 1.3.2. Thách thức về mô hình

**1. Non-linear relationships**: Delay không phải linear với bất kỳ feature nào, mà là tương tác phức tạp giữa nhiều yếu tố.

**2. High-cardinality categorical**:
- 150 drivers
- 120 trucks
- 200 customers
- 60+ routes
→ Cần encoding strategy phù hợp

**3. Temporal dependencies**:
- Seasonal patterns (Q4 peak)
- Day-of-week effects
- Hour-of-day effects
- Lag effects

**4. Imbalanced distribution**: Hầu hết trips có delay nhỏ, một số ít có delay rất lớn.

#### 1.3.3. Thách thức về đánh giá

**1. Time-based validation**: Phải tránh data leakage bằng cách split theo thời gian.

**2. Multiple metrics**: Một metric đơn không đủ → cần MAE, RMSE, R², MAPE.

**3. Business interpretation**: Cần giải thích được model cho stakeholders.

### 1.4. Ý nghĩa thực tiễn

#### 1.4.1. Đối với doanh nghiệp logistics

**Tài chính**:
- Tiết kiệm 5-10% chi phí vận hành thông qua dispatch optimization
- Tăng margin 3-5% nhờ pricing chính xác
- Giảm 20-30% chi phí detention

**Vận hành**:
- Cảnh báo sớm trips có nguy cơ delay
- Tối ưu resource allocation (driver-truck-load matching)
- Cải thiện route planning

**Khách hàng**:
- Tăng SLA compliance từ 85-95% lên 95%+
- Giảm complaints
- Tăng customer retention

#### 1.4.2. Đối với khách hàng (shippers)

- **Visibility**: Biết trước khả năng delay
- **Inventory planning**: Tránh stockout
- **Cost saving**: Giảm chi phí buffer

#### 1.4.3. Đối với toàn ngành

- Mỗi 1% giảm delay → tiết kiệm ~$740M cho ngành Mỹ
- Giảm carbon footprint do tối ưu route
- Đóng góp vào supply chain resilience

### 1.5. Phạm vi và đóng góp

**Phạm vi**:
- Dataset: Logistics Operations Database 2022-2024 (synthetic)
- Bài toán: Regression (predict delay hours)
- 3 mô hình: Random Forest, XGBoost, LightGBM
- Đánh giá: MAE, RMSE, R², MAPE

**Đóng góp**:
1. Pipeline ML hoàn chỉnh tận dụng cả 14 bảng
2. Feature engineering chi tiết với 50+ engineered features
3. So sánh có hệ thống 3 mô hình tree-based
4. Phân tích lỗi và insights nghiệp vụ
5. Source code reproducible

---

## CHƯƠNG 2: CÁC PHƯƠNG PHÁP TIẾP CẬN

### 2.1. Tổng quan các phương pháp

#### 2.1.1. Statistical / Time-series methods
- **ARIMA, SARIMA, Prophet**
- Ưu: Đơn giản, giải thích được, tốt cho pure time-series
- Nhược: Khó tích hợp nhiều features ngoài target

#### 2.1.2. Linear Models
- **Linear Regression, Ridge, Lasso, ElasticNet**
- Ưu: Nhanh, baseline, giải thích được
- Nhược: Không capture non-linearity

#### 2.1.3. Tree-based Models
- **Decision Tree**: Đơn giản nhưng dễ overfit
- **Random Forest**: Bagging, robust
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- Ưu: SOTA cho tabular, capture non-linearity
- Nhược: Khó debug khi sai, có thể overfit nếu không tune

#### 2.1.4. Support Vector Machines
- **SVR (Support Vector Regression)**
- Ưu: Mạnh với high-dim
- Nhược: Chậm với data lớn, khó tune

#### 2.1.5. Deep Learning
- **MLP, TabNet, FT-Transformer**
- **LSTM/GRU** cho sequential
- Ưu: Capture rất phức tạp
- Nhược: Cần data lớn, kém hơn tree trên tabular

#### 2.1.6. Ensemble Methods
- Stacking, Blending, Voting

### 2.2. Phương pháp được chọn

Sau khi cân nhắc, chúng tôi chọn **3 mô hình tree-based**:

1. **Random Forest Regressor** (Bagging)
2. **XGBoost Regressor** (Boosting - level-wise)
3. **LightGBM Regressor** (Boosting - leaf-wise)

#### 2.2.1. Lý do chọn tree-based methods

✅ **State-of-the-art cho tabular data** (xác nhận qua nhiều nghiên cứu và Kaggle competitions)
✅ Xử lý mixed data types tự nhiên
✅ Capture được non-linear interactions
✅ Robust với outliers
✅ Feature importance dễ giải thích
✅ Không cần feature scaling phức tạp
✅ Train nhanh, deploy được

#### 2.2.2. Random Forest Regressor

**Nguyên lý**: Tập hợp nhiều decision trees được train trên các bootstrap samples khác nhau, output là trung bình các predictions.

**Công thức**:
$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

Với $T_b$ là tree thứ $b$, $B$ là tổng số trees.

**Ưu điểm**:
- Variance reduction qua bagging
- Khó overfit với nhiều trees
- Parallel training nhanh
- Out-of-bag (OOB) score evaluation

**Vai trò trong đồ án**: Baseline mạnh để so sánh với boosting

#### 2.2.3. XGBoost Regressor

**Nguyên lý**: Gradient Boosting với regularization và optimization advanced.

**Công thức loss**:
$$L = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

Với $\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2$ là regularization term.

**Ưu điểm**:
- Built-in L1, L2 regularization
- Handle missing values
- Parallel computing
- SHAP support tốt
- Early stopping

**Vai trò trong đồ án**: Champion model, được kỳ vọng có hiệu suất cao nhất

#### 2.2.4. LightGBM Regressor

**Nguyên lý**: Gradient Boosting với leaf-wise tree growth (khác level-wise của XGBoost) và histogram-based optimization.

**Khác biệt với XGBoost**:
- **Tree growth**: Leaf-wise (chọn leaf có loss giảm nhiều nhất) vs level-wise
- **Speed**: Nhanh hơn XGBoost 3-10x
- **Memory**: Thấp hơn
- **Categorical**: Native support (không cần one-hot)

**Ưu điểm**:
- Tốc độ và memory tốt nhất
- GOSS (Gradient-based One-Side Sampling)
- EFB (Exclusive Feature Bundling)
- Native categorical handling

**Vai trò trong đồ án**: Challenger để so sánh với XGBoost

---

## CHƯƠNG 3: THỰC NGHIỆM

### 3.1. Mô tả Dataset

#### 3.1.1. Tổng quan
*[Sẽ fill sau khi EDA]*

#### 3.1.2. Phân tích target
*[Sẽ fill sau khi EDA]*

#### 3.1.3. Phân chia dữ liệu
*[Sẽ fill sau khi preprocessing]*

### 3.2. EDA
*[Sẽ fill sau khi chạy notebook 01]*

### 3.3. Tiền xử lý dữ liệu
*[Sẽ fill sau khi chạy notebook 02-03]*

### 3.4. Phương pháp đánh giá

#### 3.4.1. Metrics
- **MAE**: Sai số tuyệt đối trung bình
- **RMSE**: Penalize outliers
- **R²**: Phương sai giải thích được
- **MAPE**: Sai số phần trăm

#### 3.4.2. Validation strategy
- TimeSeriesSplit 5-fold CV
- Holdout test set (2024 data)

#### 3.4.3. Hyperparameter tuning
- Optuna - Bayesian optimization
- 50-100 trials per model

### 3.5. Kết quả đạt được

#### 3.5.1. Random Forest
*[Sẽ fill sau training]*

#### 3.5.2. XGBoost
*[Sẽ fill sau training]*

#### 3.5.3. LightGBM
*[Sẽ fill sau training]*

#### 3.5.4. So sánh tổng hợp

| Metric | Random Forest | XGBoost | LightGBM |
|--------|---------------|---------|----------|
| MAE | ? | ? | ? |
| RMSE | ? | ? | ? |
| R² | ? | ? | ? |
| MAPE | ? | ? | ? |
| Train time | ? | ? | ? |

### 3.6. Phân tích kết quả

#### 3.6.1. Model nào tốt nhất?
*[Phân tích sau experiments]*

#### 3.6.2. Phân tích metrics

#### 3.6.3. Phân tích residuals

#### 3.6.4. Feature importance

#### 3.6.5. Error analysis

#### 3.6.6. Overfitting/Underfitting

---

## CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 4.1. Kết luận

#### 4.1.1. Tóm tắt kết quả
*[Fill sau experiments]*

#### 4.1.2. Đóng góp
- Pipeline ML hoàn chỉnh tận dụng 14 bảng
- Feature engineering chi tiết
- So sánh có hệ thống 3 mô hình tree-based
- Insights nghiệp vụ

#### 4.1.3. Hạn chế
- Synthetic data
- Chưa có weather/traffic real-time
- Chưa thử deep learning

### 4.2. Hướng phát triển

**Cải thiện model**:
- Stacking ensemble
- TabNet, FT-Transformer
- AutoML (H2O, AutoGluon)

**Mở rộng features**:
- Weather data (NOAA)
- Traffic data (Google Maps)
- Economic indicators

**Cải thiện pipeline**:
- MLflow, DVC
- Real-time serving
- Monitoring & retraining

---

## TÀI LIỆU THAM KHẢO

[Xem REFERENCES/references.md]

---

## PHỤ LỤC

### A. Mã nguồn
*[SOURCE_CODE/]*

### B. Visualizations bổ sung

### C. Hyperparameters tốt nhất

### D. Bảng kết quả chi tiết
