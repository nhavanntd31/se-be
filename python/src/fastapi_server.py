from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import List, Dict, Optional
import logging
import uvicorn
import torch
import numpy as np
from model_trainning.student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    StudentPerformancePredictor,
    create_student_sequences
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_students_with_multiple_semesters(perf_df, min_semesters=3):
    """Tìm sinh viên có nhiều kỳ học nhất"""
    student_semester_counts = perf_df.groupby('student_id')['Relative Term'].count()
    qualified_students = student_semester_counts[student_semester_counts >= min_semesters]
    return qualified_students.sort_values(ascending=False).index.tolist()

def safe_float_convert(value):
    """Chuyển đổi giá trị về float an toàn"""
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def inverse_normalize_gpa_cpa(normalized_values, original_gpa_col, original_cpa_col):
    """Chuyển đổi giá trị đã normalize về giá trị gốc"""
    if original_gpa_col.empty or original_cpa_col.empty:
        return 0.0, 0.0
        
    gpa_mean = original_gpa_col.mean()
    gpa_std = original_gpa_col.std() if original_gpa_col.std() != 0 else 1.0
    cpa_mean = original_cpa_col.mean() 
    cpa_std = original_cpa_col.std() if original_cpa_col.std() != 0 else 1.0
    
    original_gpa = normalized_values[0] * gpa_std + gpa_mean
    original_cpa = normalized_values[1] * cpa_std + cpa_mean
    
    if pd.isna(original_gpa) or pd.isna(original_cpa):
        return 0.0, 0.0
        
    return original_gpa, original_cpa

def predict_next_semester(student_id, course_df, perf_df, processor, model, device):
    student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
    student_courses = course_df[course_df['student_id'] == student_id]
    total_semesters = len(student_perf)
    original_gpa_col = pd.to_numeric(perf_df['GPA'], errors='coerce').dropna()
    original_cpa_col = pd.to_numeric(perf_df['CPA'], errors='coerce').dropna()
    course_processed = processor.preprocess_course_data(course_df)
    perf_processed = processor.preprocess_performance_data(perf_df)
    student_courses_proc = course_processed[course_processed['student_id'] == student_id]
    student_perf_proc = perf_processed[perf_processed['student_id'] == student_id]
    semester_data = []
    for _, perf_row in student_perf_proc.iterrows():
        semester = perf_row['Semester']
        semester_courses = student_courses_proc[student_courses_proc['Semester'] == semester]
        if len(semester_courses) == 0:
            continue
        course_features = semester_courses[['Continuous Assessment Score', 'Exam Score', 'Credits',
                                         'Final Grade Numeric', 'Course_Category_Encoded', 
                                         'Pass_Status', 'Grade_Points']].values.tolist()
        performance_features = perf_row[['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                                       'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                                       'Debt_Rate', 'Accumulation_Rate']].values.tolist()
        semester_data.append({
            'semester': semester,
            'relative_term': perf_row['Relative Term'],
            'courses': course_features,
            'performance': performance_features
        })
    semester_data.sort(key=lambda x: x['relative_term'])
    n = len(semester_data)
    if n < 1:
        return None
    course_features = torch.zeros(1, 10, 15, 7)
    semester_features = torch.zeros(1, 10, 11)
    course_masks = torch.zeros(1, 10, 15)
    semester_masks = torch.zeros(1, 10)
    for sem_idx, sem_data in enumerate(semester_data[:9]):
        semester_features[0, sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
        semester_masks[0, sem_idx] = 1.0
        courses = sem_data['courses'][:15]
        for course_idx, course in enumerate(courses):
            course_features[0, sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
            course_masks[0, sem_idx, course_idx] = 1.0
    # slot cuối là kỳ tiếp theo (toàn 0, mask=1)
    semester_masks[0, n if n < 10 else 9] = 1.0
    # Dự đoán
    with torch.no_grad():
        prediction = model(course_features.to(device), 
                         semester_features.to(device), 
                         course_masks.to(device), 
                         semester_masks.to(device))
        pred_idx = n if n < 10 else 9
        predicted_gpa_norm = prediction[0, pred_idx, 0].item() if prediction.ndim == 3 else prediction[0, 0].item()
        predicted_cpa_norm = prediction[0, pred_idx, 1].item() if prediction.ndim == 3 else prediction[0, 1].item()
    predicted_gpa_original, predicted_cpa_original = inverse_normalize_gpa_cpa(
        [predicted_gpa_norm, predicted_cpa_norm], 
        original_gpa_col, 
        original_cpa_col
    )
    return {
        'student_id': student_id,
        'total_semesters': total_semesters,
        'predicted_next_gpa': predicted_gpa_original,
        'predicted_next_cpa': predicted_cpa_original,
        'predicted_next_gpa_normalized': predicted_gpa_norm,
        'predicted_next_cpa_normalized': predicted_cpa_norm
    }

# Khởi tạo model và processor global
device = None
model = None
processor = None

@app.on_event("startup")
async def startup_event():
    global device, model, processor
    logger.info("Initializing prediction components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        model = StudentPerformancePredictor()
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'model_trainning', 'best_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    processor = StudentPerformanceDataProcessor()
    logger.info("Prediction service initialized successfully")

@app.post("/predict-students")
async def predict_students(
    course_file: UploadFile = File(...),
    performance_file: UploadFile = File(...)
):
    try:
        logger.info(f"Received files: {course_file.filename}, {performance_file.filename}")
        
        # Đọc file CSV
        course_content = await course_file.read()
        performance_content = await performance_file.read()
        
        course_df = pd.read_csv(io.BytesIO(course_content))
        performance_df = pd.read_csv(io.BytesIO(performance_content))
        
        logger.info(f"Course data shape: {course_df.shape}, Performance data shape: {performance_df.shape}")
        
        # Tìm sinh viên có đủ dữ liệu
        qualified_students = find_students_with_multiple_semesters(performance_df, min_semesters=2)
        
        # Lọc sinh viên có cả course và performance data
        course_students = set(course_df['student_id'].unique()) if 'student_id' in course_df.columns else set()
        common_students = [s for s in qualified_students if s in course_students]
        
        logger.info(f"Found {len(common_students)} qualified students with both course and performance data")
        
        if not common_students:
            raise HTTPException(status_code=400, detail="No qualified students found with sufficient data")
        
        results = []
        
        for student_id in common_students:
            try:
                logger.info(f"Processing student: {student_id}")
                prediction_result = predict_next_semester(
                    student_id, course_df, performance_df, processor, model, device
                )
                if prediction_result:
                    results.append({
                        'student_id': student_id,
                        'prediction': prediction_result,
                        'status': 'success'
                    })
                    logger.info(f"Successfully predicted for student {student_id}")
                else:
                    results.append({
                        'student_id': student_id,
                        'prediction': None,
                        'status': 'failed',
                        'error': 'Prediction failed - insufficient data'
                    })
                    logger.warning(f"Prediction failed for student {student_id}")
            except Exception as e:
                logger.error(f"Error processing student {student_id}: {str(e)}")
                results.append({
                    'student_id': student_id,
                    'prediction': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        response = {
            'total_students': len(common_students),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'predictions': [r['prediction'] for r in successful_results]
        }
        
        logger.info(f"Completed processing. Success: {len(successful_results)}/{len(common_students)}")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in predict_students: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Student Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9005) 