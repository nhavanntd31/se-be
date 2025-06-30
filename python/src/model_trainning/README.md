# Mô Hình Dự Đoán Kết Quả Học Tập Sinh Viên

## Tổng Quan

Đây là một mô hình AI sử dụng kiến trúc **LSTM + Transformer** để dự đoán kết quả học tập (GPA và CPA) của sinh viên dựa trên lịch sử học tập từ các kỳ trước đó.

## Kiến Trúc Mô Hình

### 1. **Course-level Encoder**
- Xử lý thông tin từng môn học trong mỗi kỳ
- Input: Điểm quá trình, điểm thi, tín chỉ, loại môn học, trạng thái pass/fail
- Output: Embedding vector cho mỗi môn học

### 2. **Semester-level Aggregation** 
- Tổng hợp thông tin tất cả môn học trong kỳ thành một representation
- Sử dụng masked averaging để xử lý variable-length sequences

### 3. **Multi-Modal Fusion**
- Kết hợp thông tin từ môn học với thông tin tổng quan của kỳ (GPA, CPA, cảnh cáo, etc.)
- Fusion layer để tạo unified representation

### 4. **Transformer Encoder**
- Áp dụng self-attention để capture dependencies giữa các kỳ học
- Positional encoding để model hiểu thứ tự thời gian
- Multi-head attention để focus vào các khía cạnh khác nhau

### 5. **LSTM Layer**
- Bidirectional LSTM để học sequential patterns
- Capture long-term dependencies trong quá trình học tập

### 6. **Attention Mechanism**
- Final attention layer để chọn lọc thông tin quan trọng
- Giúp model focus vào những kỳ học có ảnh hưởng lớn đến kết quả

### 7. **Prediction Head**
- Fully connected layers với dropout
- Output: Predicted GPA và CPA cho kỳ tiếp theo

## Dữ Liệu Input

### File 1: Course Data (`ET1_K62_K63_K64 (4).csv`)
- **Semester**: Kỳ học (VD: 20171, 20172)
- **Course ID/Name**: Mã và tên môn học
- **Credits**: Số tín chỉ của môn 
- **Continuous Assessment Score**: Điểm quá trình (0-10)
- **Exam Score**: Điểm thi cuối kỳ (0-10)
- **Final Grade**: Điểm tổng kết (A+, A, B+, ..., F)
- **Relative Term**: Kỳ học thứ mấy của sinh viên
- **student_id**: Mã sinh viên

### File 2: Performance Data (`ET1_K62_K63_K64_performance (3).csv`)
- **GPA**: Điểm trung bình kỳ
- **CPA**: Điểm trung bình tích lũy
- **TC qua**: Số tín chỉ pass trong kỳ
- **Acc**: Tổng tín chỉ tích lũy
- **Debt**: Số tín chỉ nợ
- **Reg**: Số tín chỉ đăng ký
- **Warning**: Mức cảnh cáo (Mức 0, 1, 2, 3)
- **Level**: Năm học (Năm thứ nhất, hai, ba...)

## Feature Engineering

### Course-level Features:
1. **Grade Points**: Final Grade × Credits
2. **Pass Status**: Binary (pass/fail)
3. **Course Category**: Encoded course type (MI, ET, PH, etc.)
4. **Score Features**: Continuous + Exam scores

### Semester-level Features:
1. **Pass Rate**: TC qua / Reg
2. **Debt Rate**: Debt / Reg  
3. **Accumulation Rate**: Acc / (Relative Term × expected credits)
4. **Warning Level**: Numerical encoding of warning levels

## Cách Sử Dụng

### 1. Training Mô Hình
```python
python student_performance_prediction.py
```

Sẽ thực hiện:
- Load và preprocess dữ liệu
- Tạo sequences cho training
- Train mô hình với early stopping
- Lưu best model vào `best_model.pth`
- Hiển thị training plots và metrics

### 2. Demo Dự Đoán
```python
python demo_prediction.py
```

Sẽ:
- Load model đã train
- Dự đoán cho một số sinh viên mẫu
- So sánh kết quả dự đoán vs thực tế
- Phân tích trajectory học tập của sinh viên

### 3. Dự Đoán Cho Sinh Viên Cụ Thể
```python
from demo_prediction import predict_student_performance
import pandas as pd

course_df = pd.read_csv('csv/ET1_K62_K63_K64 (4).csv')
perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance (3).csv')

result = predict_student_performance(
    student_id=1, 
    course_df=course_df, 
    perf_df=perf_df
)
```

## Metrics Đánh Giá

- **MSE (Mean Squared Error)**: Sai số bình phương trung bình
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình  
- **R² Score**: Hệ số xác định (0-1, càng cao càng tốt)

## Hyperparameters

```python
- hidden_dim: 64          # Dimension của hidden representations
- lstm_hidden: 128        # LSTM hidden size
- num_heads: 8           # Số attention heads
- max_semesters: 10      # Số kỳ tối đa trong sequence
- max_courses: 15        # Số môn tối đa mỗi kỳ
- batch_size: 32         # Training batch size
- learning_rate: 0.001   # Learning rate
- dropout: 0.1-0.2       # Dropout rates
```

## Ưu Điểm

1. **Xử lý Variable Length**: Model có thể xử lý sinh viên với số kỳ học và số môn/kỳ khác nhau
2. **Multi-Modal**: Kết hợp cả thông tin chi tiết môn học và thông tin tổng quan
3. **Attention Mechanism**: Tự động identify những kỳ học quan trọng nhất
4. **Sequential Learning**: LSTM capture temporal dependencies
5. **Robust**: Dropout và regularization để tránh overfitting

## Hạn Chế

1. **Data Dependency**: Cần đủ lịch sử học tập để dự đoán chính xác
2. **Domain Specific**: Model được train specific cho chương trình Điện tử - Viễn thông
3. **Cold Start**: Khó dự đoán cho sinh viên mới không có lịch sử

## Kết Quả Mong Đợi

- **GPA Error**: < 0.3 điểm trung bình
- **CPA Error**: < 0.2 điểm trung bình
- **R² Score**: > 0.7 cho cả GPA và CPA

## Requirements

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## File Structure

```
├── student_performance_prediction.py  # Main model và training
├── demo_prediction.py                 # Demo script
├── README.md                          # Documentation
├── csv/
│   ├── ET1_K62_K63_K64 (4).csv       # Course data
│   └── ET1_K62_K63_K64_performance (3).csv  # Performance data
├── best_model.pth                     # Saved model weights
└── training_results.png              # Training visualization
```

## Mở Rộng

1. **Thêm Features**: Course difficulty, instructor rating, time of day
2. **Ensemble Methods**: Combine multiple models
3. **Multi-Task Learning**: Predict other outcomes (graduation time, career success)
4. **Real-time Updates**: Online learning để update model với dữ liệu mới
5. **Explainability**: SHAP values để hiểu model decisions
