# OUTLINE SLIDE THUYẾT TRÌNH

> **Đề tài**: Dự đoán độ trễ giao hàng trong vận tải đường bộ
> **Thời lượng**: 15-20 phút trình bày + Q&A
> **Số slide**: 18-22 slides

---

## Slide 1: TITLE
- **Tên đề tài**: Dự đoán độ trễ giao hàng (Trip Delay Prediction) trong vận tải đường bộ
- **Tên tiếng Anh**: Trip Delay Hours Prediction in Road Transportation
- Sinh viên: Trần Tín Nghĩa
- MSSV, Lớp, Trường UIT
- GVHD: ...

## Slide 2: NỘI DUNG TRÌNH BÀY
1. Giới thiệu đề tài
2. Dataset
3. Phương pháp tiếp cận
4. Pipeline thực hiện
5. Kết quả thực nghiệm
6. Kết luận & Hướng phát triển

---

## PHẦN 1: GIỚI THIỆU (Slides 3-5)

### Slide 3: ĐẶT VẤN ĐỀ
- Ngành logistics: $940B/năm tại Mỹ
- Delay gây thiệt hại: $74B/năm
- Tác động: SLA, customer, supply chain
- **Hình ảnh**: Infographic ngành logistics

### Slide 4: BÀI TOÁN
- **Input**: Thông tin chuyến hàng (driver, truck, route, customer, facility...)
- **Output**: Số giờ delay (regression)
- **Loại**: Supervised Learning - Regression
- **Hình ảnh**: Diagram input → model → output

### Slide 5: KHÓ KHĂN & THÁCH THỨC
- 14 bảng dữ liệu phức tạp
- 361K+ records
- Mixed data types
- Non-linear relationships
- Temporal dependencies
- High-cardinality categorical

### Slide 6: Ý NGHĨA THỰC TIỄN
- Tiết kiệm 5-10% chi phí
- Tăng SLA compliance lên 95%+
- ROI rõ ràng và đo lường được
- **Hình ảnh**: ROI chart

---

## PHẦN 2: DATASET (Slides 7-8)

### Slide 7: TỔNG QUAN DATASET
- Logistics Operations Database (Kaggle)
- 14 bảng, 361,799 records
- 57,096 trips
- Thời gian 2022-2024
- **Hình ảnh**: ER diagram đơn giản

### Slide 8: PHÂN TÍCH TARGET
- Distribution của delay_hours
- Mean, median, std
- Outliers
- **Biểu đồ**: Histogram + boxplot

---

## PHẦN 3: PHƯƠNG PHÁP (Slides 9-11)

### Slide 9: TỔNG QUAN CÁC PHƯƠNG PHÁP
- Statistical/Time-series
- Linear models
- **Tree-based** ← Chọn
- Deep learning
- Ensemble
- **Bảng so sánh ưu/nhược điểm**

### Slide 10: 3 MÔ HÌNH ĐƯỢC CHỌN
| Model | Loại | Đặc điểm |
|-------|------|----------|
| Random Forest | Bagging | Baseline, robust |
| XGBoost | Boosting (level) | Champion, SOTA |
| LightGBM | Boosting (leaf) | Fast, native cat |

### Slide 11: LÝ DO CHỌN
- ✅ SOTA cho tabular data
- ✅ Capture non-linearity
- ✅ Mixed data types
- ✅ Feature importance
- ✅ Robust với outliers

---

## PHẦN 4: PIPELINE (Slides 12-14)

### Slide 12: PIPELINE TỔNG THỂ
```
Raw Data (14 tables)
    ↓
EDA & Cleaning
    ↓
Feature Engineering (50+ features)
    ↓
Time-based Split (70/15/15)
    ↓
Train 3 Models
    ↓
Evaluation & Comparison
    ↓
Best Model
```

### Slide 13: EDA HIGHLIGHTS
- 3-4 visualizations đặc sắc nhất
- Insights chính

### Slide 14: FEATURE ENGINEERING
- Driver features (tenure, on-time history)
- Truck features (age, maintenance)
- Route features (distance, historical)
- Temporal features (month, dow, hour)
- Customer features
- Lag features

---

## PHẦN 5: KẾT QUẢ (Slides 15-19)

### Slide 15: PHƯƠNG PHÁP ĐÁNH GIÁ
- **Metrics**: MAE, RMSE, R², MAPE
- **Validation**: TimeSeriesSplit 5-fold
- **Tuning**: Optuna 50-100 trials
- **Test**: 2024 data (holdout)

### Slide 16: KẾT QUẢ 3 MODELS
| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|----|
| RF | ? | ? | ? | ? |
| XGB | ? | ? | ? | ? |
| LGB | ? | ? | ? | ? |

### Slide 17: SO SÁNH TRỰC QUAN
- **Bar chart** comparing metrics
- **Residual plots** side-by-side

### Slide 18: FEATURE IMPORTANCE
- **SHAP plot** từ best model
- Top 10 features
- Insights nghiệp vụ

### Slide 19: ERROR ANALYSIS
- Trips dự đoán sai nhiều
- Patterns of errors
- Train vs Val curves

---

## PHẦN 6: KẾT LUẬN (Slides 20-22)

### Slide 20: KẾT LUẬN
- Best model: ?
- MAE: ? giờ
- R²: ?
- Đóng góp:
  - ✅ Pipeline tận dụng 14 bảng
  - ✅ 50+ engineered features
  - ✅ So sánh có hệ thống

### Slide 21: HƯỚNG PHÁT TRIỂN
- Stacking ensemble
- TabNet, FT-Transformer
- Tích hợp weather/traffic
- MLOps deployment

### Slide 22: CẢM ƠN & Q&A
- Cảm ơn GVHD và hội đồng
- Q&A
- Contact info

---

## 🎨 LƯU Ý DESIGN

### Phong cách
- Professional, minimal
- Màu chủ đạo: xanh navy + cam (logistics theme)
- Font: Sans-serif (Calibri, Arial)
- Avoid: Quá nhiều text trên một slide

### Mỗi slide
- Tiêu đề rõ ràng
- 3-5 bullet points
- 1 hình/biểu đồ
- Source nếu cần

### Visualizations cần có
1. ER diagram dataset
2. Target distribution
3. Pipeline flowchart
4. Feature importance (SHAP)
5. Metrics comparison bar chart
6. Residual plots
7. Train vs Val curves

### Tips trình bày
- Mở đầu bằng câu hook về business value
- Kể câu chuyện: từ vấn đề → giải pháp → kết quả
- Demo nếu có thể (notebook results)
- Chuẩn bị Q&A về:
  - Vì sao chọn 3 model này?
  - Vì sao không dùng deep learning?
  - Làm sao tránh data leakage?
  - Business value cụ thể?
