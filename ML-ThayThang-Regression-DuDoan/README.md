# Đồ án Máy học - Trip Delay Prediction

## 📋 Thông tin

- **Đề tài**: Dự đoán độ trễ giao hàng (Trip Delay Hours Prediction) trong vận tải đường bộ
- **Bài toán**: Regression
- **Sinh viên**: Trần Tín Nghĩa - Hồ Thị Mỹ Phương - Trương Ngọc Sơn
- **Trường**: UIT - Đại học Công nghệ Thông tin
- **Môn**: Máy học
- **Năm học**: 2025-2026

## 🎯 Mục tiêu

Dự đoán số giờ delay của một chuyến hàng trong logistics, sử dụng 3 mô hình:
1. **Random Forest** (Bagging)
2. **XGBoost** (Boosting - level-wise)
3. **LightGBM** (Boosting - leaf-wise)

## 📁 Cấu trúc Project

```
DoAnMonHoc/
├── PLAN.md                       # Kế hoạch tổng thể
├── README.md                     # File này
│
├── DOCS/                         # 📄 Báo cáo Word
│   ├── BaoCao_Template.md        # Template báo cáo
│   ├── outline.md                # Outline chi tiết
│   ├── BaoCao_DoAn.docx          # File Word cuối (sẽ tạo sau)
│   └── figures/                  # Hình ảnh
│
├── SLIDES/                       # 🎤 Slide trình chiếu
│   ├── outline.md
│   └── Slides_DoAn.pptx          # File PPT (sẽ tạo sau)
│
├── SOURCE_CODE/                  # 💻 Source code
│   ├── notebooks/                # Jupyter notebooks
│   ├── src/                      # Python modules
│   ├── data/                     # Data
│   ├── models/                   # Saved models
│   ├── results/                  # Kết quả
│   └── requirements.txt
│
└── REFERENCES/                   # 📚 Tài liệu tham khảo
    └── references.md
```

## 🚀 Cài đặt

```bash
cd SOURCE_CODE
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## 📊 Dataset

- **Nguồn**: [Kaggle - Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database)
- **Tải về**: Đặt vào `SOURCE_CODE/data/raw/`

```bash
# Download via Kaggle CLI
cd SOURCE_CODE/data/raw
kaggle datasets download -d yogape/logistics-operations-database
unzip logistics-operations-database.zip
```

## ▶️ Cách chạy

Chạy notebooks theo thứ tự:

```bash
cd SOURCE_CODE
jupyter lab
```

Sau đó mở các notebook trong `notebooks/`:
1. `01_EDA.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_Feature_Engineering.ipynb`
4. `04_Modeling_RandomForest.ipynb`
5. `05_Modeling_XGBoost.ipynb`
6. `06_Modeling_LightGBM.ipynb`
7. `07_Comparison_Analysis.ipynb`

## 📈 Kết quả mong đợi

Comparison sẽ được tổng hợp trong notebook 07:

| Model | MAE (h) | RMSE (h) | R² | MAPE | Train Time |
|-------|---------|----------|----|----|------------|
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |

## 📝 Sản phẩm nộp

- [x] **DOCS/BaoCao_DoAn.docx** - Báo cáo Word đầy đủ 10 mục
- [x] **SLIDES/Slides_DoAn.pptx** - Slide thuyết trình
- [x] **SOURCE_CODE/** - Code đầy đủ
- [x] **README.md** - Hướng dẫn

## 📚 Tài liệu tham khảo

Xem `REFERENCES/references.md`
