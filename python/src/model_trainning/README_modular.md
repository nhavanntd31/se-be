# Student Performance Prediction - Modular Version

Đây là phiên bản modular của hệ thống dự đoán kết quả học tập sinh viên, được tách thành các file riêng biệt để dễ quản lý và so sánh.

## Cấu trúc file

```
├── data_processor.py          # Data processing chung cho cả ML và DL
├── predict_ml.py             # Traditional ML models (RF, GB, Linear, Ridge, Lasso, SVM)
├── train_transformer.py     # Transformer/Deep Learning model
└── README_modular.md        # File hướng dẫn này
```

## File descriptions

### 1. `data_processor.py`
- **Mục đích**: Xử lý dữ liệu chung cho cả traditional ML và deep learning
- **Chức năng chính**:
  - `StudentPerformanceDataProcessor`: Class xử lý và normalize dữ liệu
  - `create_student_sequences()`: Tạo sequences từ dữ liệu course và performance
  - `compute_denormalized_metrics()`: Tính metrics trên scale gốc (denormalized)
  - `load_and_prepare_data()`: Helper function load và chuẩn bị dữ liệu

### 2. `predict_ml.py`
- **Mục đích**: Training và evaluation traditional ML models
- **Models**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVM
- **Output**: 
  - Best GPA và CPA models (`.pkl` files)
  - Comparison plot (`ml_algorithms_comparison.png`)
  - Performance metrics (normalized và denormalized)

### 3. `train_transformer.py`
- **Mục đích**: Training và evaluation deep learning model
- **Architecture**: Course Encoder + Transformer + LSTM + Attention
- **Output**:
  - Best model weights (`best_model.pth`)
  - Training plots (`transformer_results.png`)
  - Performance metrics (normalized và denormalized)

## Cách sử dụng

### 1. Chạy Traditional ML Models

```bash
python predict_ml.py --course_csv csv/ET1_K62_K63_K64.csv --performance_csv csv/ET1_K62_K63_K64_performance.csv
```

**Output:**
- `best_gpa_model.pkl`: Model tốt nhất cho GPA prediction
- `best_cpa_model.pkl`: Model tốt nhất cho CPA prediction
- `ml_algorithms_comparison.png`: So sánh performance các models
- `feature_scaler.pkl`, `course_scaler.pkl`: Scalers để denormalize

### 2. Chạy Transformer Model

```bash
python train_transformer.py --course_csv csv/ET1_K62_K63_K64.csv --performance_csv csv/ET1_K62_K63_K64_performance.csv --epochs 50
```

**Options:**
- `--epochs`: Số epochs training (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--output_model`: Path lưu model weights (default: best_model.pth)
- `--output_plot`: Path lưu training plots (default: transformer_results.png)

**Output:**
- `best_model.pth`: Model weights tốt nhất
- `transformer_results.png`: Training progress plots
- `feature_scaler.pkl`, `course_scaler.pkl`: Scalers để denormalize

### 3. So sánh kết quả

Sau khi chạy cả hai, bạn có thể so sánh:

#### Traditional ML Results:
```
Model                GPA MAE    GPA MSE    GPA R²     CPA MAE    CPA MSE    CPA R²     Time(s)
Random Forest        0.1234     0.0567     0.8901     0.1345     0.0678     0.8765     12.34
Gradient Boosting    0.1145     0.0512     0.9012     0.1234     0.0589     0.8890     25.67
...
```

#### Transformer Results:
```
=== RESULTS ON ORIGINAL SCALE (DENORMALIZED) ===
GPA Prediction:
  - MSE: 0.0456
  - MAE: 0.1089
  - R²: 0.9234

CPA Prediction:
  - MSE: 0.0534
  - MAE: 0.1156
  - R²: 0.9123
```

## Ưu điểm của kiến trúc modular

### 1. **Dễ quản lý**
- Mỗi file có chức năng riêng biệt
- Code rõ ràng, dễ debug
- Dễ maintain và update

### 2. **Dễ so sánh**
- Chạy từng loại model riêng biệt
- So sánh performance một cách khách quan
- Metrics đồng nhất (denormalized) cho tất cả models

### 3. **Tái sử dụng**
- `data_processor.py` dùng chung cho mọi model
- Scalers được save/load consistently
- Có thể extend thêm models mới dễ dàng

### 4. **Linh hoạt**
- Có thể chạy chỉ ML hoặc chỉ DL
- Customize hyperparameters cho từng loại model
- Output format nhất quán

## Key Features

### 1. **Denormalized Metrics**
- Tất cả metrics được tính trên original scale (GPA 0-4, CPA 0-4)
- Dễ interpret và so sánh với actual performance

### 2. **Robust Data Processing**
- Handle missing values, outliers
- Consistent normalization/denormalization
- Student dropout detection

### 3. **Comprehensive Evaluation**
- MSE, MAE, R² cho cả GPA và CPA
- Training plots và comparison charts
- Best model selection tự động

### 4. **Production Ready**
- Save/load models và scalers
- Error handling và logging
- GPU support cho deep learning

## Requirements

```
pandas
numpy
torch
scikit-learn
matplotlib
seaborn
```

## Example Workflow

```bash
# 1. Chạy traditional ML để có baseline
python predict_ml.py --course_csv data.csv --performance_csv perf.csv

# 2. Chạy transformer model
python train_transformer.py --course_csv data.csv --performance_csv perf.csv --epochs 100

# 3. So sánh kết quả và chọn model tốt nhất
# - Xem ml_algorithms_comparison.png
# - Xem transformer_results.png
# - So sánh denormalized MAE values
```

Kiến trúc này giúp bạn dễ dàng:
- Thử nghiệm các approaches khác nhau
- So sánh performance một cách công bằng
- Deploy model tốt nhất cho production
- Maintain và update code dễ dàng 