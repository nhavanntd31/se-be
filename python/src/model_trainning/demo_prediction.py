import pandas as pd
import numpy as np
import torch
from .student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    StudentPerformancePredictor,
    create_student_sequences
)
import pickle

def predict_student_performance(student_id: int, course_df: pd.DataFrame, 
                              perf_df: pd.DataFrame, model_path: str = 'best_model.pth'):
    print(f"=== DỰ ĐOÁN KẾT QUẢ HỌC TẬP CHO SINH VIÊN {student_id} ===")
    
    processor = StudentPerformanceDataProcessor()
    sequences = create_student_sequences(course_df, perf_df, processor)
    
    student_sequences = [seq for seq in sequences if seq['student_id'] == student_id]
    
    if not student_sequences:
        print(f"Không tìm thấy dữ liệu cho sinh viên {student_id}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StudentPerformancePredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    latest_sequence = student_sequences[-1]
    
    course_features = torch.zeros(1, 10, 15, 7)
    semester_features = torch.zeros(1, 10, 11)
    course_masks = torch.zeros(1, 10, 15)
    semester_masks = torch.zeros(1, 10)
    
    for sem_idx, sem_data in enumerate(latest_sequence['semesters'][:10]):
        semester_features[0, sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
        semester_masks[0, sem_idx] = 1.0
        
        courses = sem_data['courses'][:15]
        for course_idx, course in enumerate(courses):
            course_features[0, sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
            course_masks[0, sem_idx, course_idx] = 1.0
    
    with torch.no_grad():
        prediction = model(course_features, semester_features, course_masks, semester_masks)
        predicted_gpa = prediction[0, 0].item()
        predicted_cpa = prediction[0, 1].item()
    
    print(f"\nLịch sử học tập ({len(latest_sequence['semesters'])} kỳ):")
    for i, sem in enumerate(latest_sequence['semesters']):
        gpa = sem['performance'][0]
        cpa = sem['performance'][1]
        courses_count = len(sem['courses'])
        print(f"  Kỳ {i+1}: GPA={gpa:.2f}, CPA={cpa:.2f}, Số môn={courses_count}")
    
    print(f"\nDỰ ĐOÁN KỲ TIẾP THEO:")
    print(f"  GPA dự đoán: {predicted_gpa:.2f}")
    print(f"  CPA dự đoán: {predicted_cpa:.2f}")
    
    actual_gpa = latest_sequence['target_gpa']
    actual_cpa = latest_sequence['target_cpa']
    
    print(f"\nKẾT QUẢ THỰC TẾ (để so sánh):")
    print(f"  GPA thực tế: {actual_gpa:.2f}")
    print(f"  CPA thực tế: {actual_cpa:.2f}")
    
    gpa_error = abs(predicted_gpa - actual_gpa)
    cpa_error = abs(predicted_cpa - actual_cpa)
    
    print(f"\nSAI SỐ:")
    print(f"  GPA error: {gpa_error:.3f}")
    print(f"  CPA error: {cpa_error:.3f}")
    
    return {
        'predicted_gpa': predicted_gpa,
        'predicted_cpa': predicted_cpa,
        'actual_gpa': actual_gpa,
        'actual_cpa': actual_cpa,
        'gpa_error': gpa_error,
        'cpa_error': cpa_error
    }

def analyze_student_trajectory(student_id: int, course_df: pd.DataFrame, perf_df: pd.DataFrame):
    print(f"\n=== PHÂN TÍCH QUÁ TRÌNH HỌC TẬP SINH VIÊN {student_id} ===")
    
    student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
    student_courses = course_df[course_df['student_id'] == student_id]
    
    if len(student_perf) == 0:
        print(f"Không tìm thấy dữ liệu cho sinh viên {student_id}")
        return
    
    print("\nTiến độ GPA/CPA theo kỳ:")
    for _, row in student_perf.iterrows():
        semester = row['Semester']
        gpa = row['GPA'] if pd.notna(row['GPA']) else 0
        cpa = row['CPA'] if pd.notna(row['CPA']) else 0
        warning = row['Warning']
        tc_qua = row['TC qua'] if pd.notna(row['TC qua']) else 0
        
        semester_courses = student_courses[student_courses['Semester'] == semester]
        passed_courses = len(semester_courses[semester_courses['Final Grade'].isin(['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D'])])
        total_courses = len(semester_courses)
        
        print(f"  Kỳ {row['Relative Term']}: GPA={gpa:.2f}, CPA={cpa:.2f}, "
              f"TC qua={tc_qua}, Pass rate={passed_courses}/{total_courses}, {warning}")
    
    avg_gpa = student_perf['GPA'].mean()
    final_cpa = student_perf['CPA'].iloc[-1] if pd.notna(student_perf['CPA'].iloc[-1]) else 0
    
    print(f"\nTÓM TẮT:")
    print(f"  GPA trung bình: {avg_gpa:.2f}")
    print(f"  CPA cuối cùng: {final_cpa:.2f}")
    print(f"  Số kỳ học: {len(student_perf)}")
    
    weak_semesters = student_perf[student_perf['GPA'] < 2.0]
    if len(weak_semesters) > 0:
        print(f"  Số kỳ yếu (GPA < 2.0): {len(weak_semesters)}")
    
    improvement_trend = student_perf['CPA'].diff().dropna()
    if len(improvement_trend) > 0:
        avg_improvement = improvement_trend.mean()
        print(f"  Xu hướng cải thiện CPA: {avg_improvement:+.3f} điểm/kỳ")

def demo_multiple_students():
    print("Đang tải dữ liệu...")
    course_df = pd.read_csv('csv/ET1_K62_K63_K64.csv')
    perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
    
    unique_students = course_df['student_id'].unique()[:5]
    
    print(f"Demo với {len(unique_students)} sinh viên đầu tiên...")
    
    results = []
    
    for student_id in unique_students:
        try:
            result = predict_student_performance(student_id, course_df, perf_df)
            if result:
                results.append(result)
            
            analyze_student_trajectory(student_id, course_df, perf_df)
            print("-" * 70)
            
        except Exception as e:
            print(f"Lỗi khi xử lý sinh viên {student_id}: {e}")
            continue
    
    if results:
        print("\n=== TỔNG KẾT HIỆU SUẤT MÔ HÌNH ===")
        avg_gpa_error = np.mean([r['gpa_error'] for r in results])
        avg_cpa_error = np.mean([r['cpa_error'] for r in results])
        
        print(f"Trung bình sai số GPA: {avg_gpa_error:.3f}")
        print(f"Trung bình sai số CPA: {avg_cpa_error:.3f}")
        
        accurate_gpa_predictions = sum(1 for r in results if r['gpa_error'] < 0.3)
        accurate_cpa_predictions = sum(1 for r in results if r['cpa_error'] < 0.3)
        
        print(f"Dự đoán GPA chính xác (error < 0.3): {accurate_gpa_predictions}/{len(results)} ({accurate_gpa_predictions/len(results)*100:.1f}%)")
        print(f"Dự đoán CPA chính xác (error < 0.3): {accurate_cpa_predictions}/{len(results)} ({accurate_cpa_predictions/len(results)*100:.1f}%)")

def load_model_and_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StudentPerformancePredictor()
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    with open('feature_scaler.pkl', 'rb') as f:
        performance_scaler = pickle.load(f)
    
    print("Model và scaler đã được load thành công!")
    print(f"Device: {device}")
    
    dummy_course_features = torch.randn(1, 10, 15, 7).to(device)
    dummy_semester_features = torch.randn(1, 10, 11).to(device)
    dummy_course_masks = torch.ones(1, 10, 15).to(device)
    dummy_semester_masks = torch.ones(1, 10).to(device)
    
    with torch.no_grad():
        predictions = model(dummy_course_features, dummy_semester_features, 
                          dummy_course_masks, dummy_semester_masks)
        
        gpa_pred_norm = predictions[0, 0].cpu().numpy()
        cpa_pred_norm = predictions[0, 1].cpu().numpy()
        
        print(f"\nKết quả dự đoán (normalized):")
        print(f"GPA: {gpa_pred_norm:.4f}")
        print(f"CPA: {cpa_pred_norm:.4f}")
        
        dummy_features = np.zeros((1, 11))
        dummy_features[0, 0] = gpa_pred_norm
        dummy_features[0, 1] = cpa_pred_norm
        
        denorm_features = performance_scaler.inverse_transform(dummy_features)
        gpa_pred_denorm = denorm_features[0, 0]
        cpa_pred_denorm = denorm_features[0, 1]
        
        print(f"\nKết quả dự đoán (denormalized - scale gốc):")
        print(f"GPA: {gpa_pred_denorm:.4f}")
        print(f"CPA: {cpa_pred_denorm:.4f}")
        
        print(f"\nLưu ý: Đây là demo với dữ liệu ngẫu nhiên")
        print(f"Để dự đoán thực tế, cần truyền vào dữ liệu course và semester thực")

if __name__ == "__main__":
    demo_multiple_students()
    load_model_and_predict() 