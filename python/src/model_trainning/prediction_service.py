import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os
import logging
from .student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    StudentPerformancePredictor,
    create_student_sequences
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_float_convert(value):
    """Chuyển đổi giá trị về float an toàn"""
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def inverse_normalize_gpa_cpa(normalized_values, original_gpa_col, original_cpa_col):
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

class StudentPredictionService:
    def __init__(self, model_path: str = None):
        logger.info("Initializing StudentPredictionService...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.processor = StudentPerformanceDataProcessor()
        logger.info("Data processor initialized")
        
        self.model = StudentPerformancePredictor()
        logger.info("Model architecture initialized")
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info("Model loaded successfully and set to evaluation mode")
        else:
            logger.warning(f"Model file not found at {model_path}")
        
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
    
    def predict_student_next_semester(self, 
                                    student_courses: List[Dict],
                                    student_performance: List[Dict]) -> Optional[Dict]:
        logger.info("Starting prediction for student next semester")
        logger.info(f"Input data - Courses: {len(student_courses)} records, Performance: {len(student_performance)} records")
        
        try:
            if len(student_courses) == 0 or len(student_performance) == 0:
                logger.warning("Empty input data - no courses or performance records")
                return None
            
            logger.info("Converting input data to DataFrames")
            course_df = pd.DataFrame(student_courses)
            perf_df = pd.DataFrame(student_performance)
            logger.info(f"DataFrames created - Course shape: {course_df.shape}, Performance shape: {perf_df.shape}")
            
            student_id = perf_df['student_id'].iloc[0] if 'student_id' in perf_df.columns else 'unknown'
            logger.info(f"Processing student_id: {student_id}")
            
            # Lưu giá trị GPA/CPA gốc để inverse normalize
            original_gpa_col = pd.to_numeric(perf_df['GPA'], errors='coerce').dropna()
            original_cpa_col = pd.to_numeric(perf_df['CPA'], errors='coerce').dropna()
            
            # Xử lý dữ liệu
            logger.info("Processing course and performance data")
            course_processed = self.processor.preprocess_course_data(course_df)
            perf_processed = self.processor.preprocess_performance_data(perf_df)
            
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
            
            if len(semester_data) < 1:
                logger.warning("No valid semester data found")
                return None
            
            logger.info(f"Processing {len(semester_data)} semesters of data")
            
            # Tạo tensor input
            course_features = torch.zeros(1, 10, 15, 7)
            semester_features = torch.zeros(1, 10, 11)
            course_masks = torch.zeros(1, 10, 15)
            semester_masks = torch.zeros(1, 10)
            
            semesters_used = 0
            courses_used = 0
            
            for sem_idx, sem_data in enumerate(semester_data[:10]):
                semester_features[0, sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
                semester_masks[0, sem_idx] = 1.0
                semesters_used += 1
                
                courses = sem_data['courses'][:15]
                for course_idx, course in enumerate(courses):
                    course_features[0, sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
                    course_masks[0, sem_idx, course_idx] = 1.0
                    courses_used += 1
            
            logger.info(f"Prepared tensors - Semesters: {semesters_used}, Total courses: {courses_used}")
            
            # Dự đoán
            logger.info("Running model prediction")
            with torch.no_grad():
                course_features = course_features.to(self.device)
                semester_features = semester_features.to(self.device)
                course_masks = course_masks.to(self.device)
                semester_masks = semester_masks.to(self.device)
                
                prediction = self.model(course_features, semester_features, course_masks, semester_masks)
                predicted_gpa_norm = prediction[0, 0].item()
                predicted_cpa_norm = prediction[0, 1].item()
                
                logger.info(f"Raw normalized output - GPA: {predicted_gpa_norm:.4f}, CPA: {predicted_cpa_norm:.4f}")
            
            # Chuyển đổi dự đoán về giá trị gốc
            if len(original_gpa_col) > 0 and len(original_cpa_col) > 0:
                predicted_gpa_original, predicted_cpa_original = inverse_normalize_gpa_cpa(
                    [predicted_gpa_norm, predicted_cpa_norm], 
                    original_gpa_col, 
                    original_cpa_col
                )
                logger.info(f"Denormalized predictions - GPA: {predicted_gpa_original:.4f}, CPA: {predicted_cpa_original:.4f}")
            else:
                logger.warning("No original data for denormalization, using normalized values")
                predicted_gpa_original = predicted_gpa_norm
                predicted_cpa_original = predicted_cpa_norm
            
            result = {
                'predicted_gpa': predicted_gpa_original,
                'predicted_cpa': predicted_cpa_original,
                'predicted_gpa_normalized': predicted_gpa_norm,
                'predicted_cpa_normalized': predicted_cpa_norm,
                'semesters_analyzed': semesters_used,
                'student_id': student_id
            }
            
            logger.info(f"Prediction completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            return None
    
    def predict_multiple_scenarios(self,
                                 student_courses: List[Dict],
                                 student_performance: List[Dict]) -> Optional[Dict]:
        logger.info("Starting multiple scenarios prediction")
        
        try:
            base_prediction = self.predict_student_next_semester(student_courses, student_performance)
            
            if not base_prediction:
                logger.warning("Base prediction failed, cannot generate scenarios")
                return None
            
            latest_perf = student_performance[-1] if student_performance else {}
            current_cpa = safe_float_convert(latest_perf.get('CPA', 0))
            logger.info(f"Current CPA for scenarios: {current_cpa}")
            
            scenarios = {
                'base_prediction': base_prediction,
                'current_cpa': current_cpa,
                'scenarios': {
                    'excellent_performance': {
                        'description': 'Nếu đạt điểm xuất sắc (GPA 3.5+)',
                        'estimated_gpa': min(4.0, base_prediction['predicted_gpa'] + 0.5),
                        'estimated_cpa': min(4.0, current_cpa + 0.1)
                    },
                    'good_performance': {
                        'description': 'Nếu đạt điểm khá (GPA 2.5-3.5)',
                        'estimated_gpa': max(2.5, min(3.5, base_prediction['predicted_gpa'])),
                        'estimated_cpa': current_cpa + 0.05
                    },
                    'poor_performance': {
                        'description': 'Nếu học kém (GPA < 2.0)',
                        'estimated_gpa': min(2.0, base_prediction['predicted_gpa'] - 0.3),
                        'estimated_cpa': max(0, current_cpa - 0.1)
                    }
                }
            }
            
            logger.info("Multiple scenarios generated successfully")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error in scenario prediction: {str(e)}", exc_info=True)
            return None

def create_prediction_service(model_path: str = None) -> StudentPredictionService:
    logger.info("Creating new prediction service instance")
    return StudentPredictionService(model_path)
