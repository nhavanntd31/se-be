# 🚀 Quick Start Guide

## Cách Sử Dụng Nhanh

### 1. **Cài Đặt Dependencies**
```bash
cd python/src/model_trainning
python run.py --mode install
```

### 2. **Phân Tích Dữ Liệu** 
```bash
python run.py --mode analyze
```

### 3. **Train Mô Hình**
```bash
python run.py --mode train
```

### 4. **Chạy Demo**
```bash
python run.py --mode demo
```

### 5. **Chạy Toàn Bộ Pipeline**
```bash
python run.py --mode all
```

## File Structure

```
model_trainning/
├── student_performance_prediction.py  # 🧠 Main model + training
├── demo_prediction.py                 # 🎬 Demo predictions  
├── data_analysis.py                   # 📊 Data analysis
├── install_dependencies.py            # 📦 Setup script
├── run.py                             # 🚀 Main runner
├── requirements.txt                   # 📋 Dependencies
├── README.md                          # 📖 Full documentation
├── QUICKSTART.md                      # ⚡ This file
└── __init__.py                        # 📁 Package init
```

## Expected Outputs

- `best_model.pth` - Trained model weights
- `training_results.png` - Training visualization  
- `data_analysis_plots.png` - Data analysis plots

## Architecture

**Course Encoder** → **Transformer** → **LSTM** → **Attention** → **Predictions**

## Performance Targets

- 📈 **GPA Error**: < 0.3 points
- 📈 **CPA Error**: < 0.2 points  
- 📈 **R² Score**: > 0.7

---
🎓 **Happy Predicting!** 