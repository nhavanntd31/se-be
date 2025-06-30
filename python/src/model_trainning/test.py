import pandas as pd
import numpy as np
import torch
from student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    StudentPerformancePredictor,
    create_student_sequences
)

def find_students_with_multiple_semesters(perf_df, min_semesters=3):
    """Tìm sinh viên có nhiều kỳ học nhất"""
    student_semester_counts = perf_df.groupby('student_id')['Relative Term'].count()
    qualified_students = student_semester_counts[student_semester_counts >= min_semesters]
    return qualified_students.sort_values(ascending=False).head(10).index.tolist()

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
    # Tính mean và std từ dữ liệu gốc
    gpa_mean = original_gpa_col.mean()
    gpa_std = original_gpa_col.std()
    cpa_mean = original_cpa_col.mean()
    cpa_std = original_cpa_col.std()
    
    # Inverse normalize
    original_gpa = normalized_values[0] * gpa_std + gpa_mean
    original_cpa = normalized_values[1] * cpa_std + cpa_mean
    
    return original_gpa, original_cpa

def predict_final_semester(student_id, course_df, perf_df, processor, model, device):
    """Dự đoán kỳ cuối cùng dựa trên n-1 kỳ trước"""
    student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
    student_courses = course_df[course_df['student_id'] == student_id]
    
    total_semesters = len(student_perf)
    
    # Lưu giá trị GPA/CPA gốc để inverse normalize
    original_gpa_col = pd.to_numeric(perf_df['GPA'], errors='coerce').dropna()
    original_cpa_col = pd.to_numeric(perf_df['CPA'], errors='coerce').dropna()
    
    # Xử lý dữ liệu
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
    
    if len(semester_data) < 2:
        return None
    
    # Sử dụng n-1 kỳ để dự đoán kỳ thứ n
    input_semesters = semester_data[:-1]  # n-1 kỳ
    target_semester = semester_data[-1]   # kỳ thứ n
    
    # Lấy giá trị GPA/CPA thực tế của kỳ cuối cùng
    target_semester_info = student_perf.iloc[-1]
    actual_gpa_original = safe_float_convert(target_semester_info['GPA'])
    actual_cpa_original = safe_float_convert(target_semester_info['CPA'])
    
    # Tạo tensor input
    course_features = torch.zeros(1, 10, 15, 7)
    semester_features = torch.zeros(1, 10, 11)
    course_masks = torch.zeros(1, 10, 15)
    semester_masks = torch.zeros(1, 10)
    
    for sem_idx, sem_data in enumerate(input_semesters[:10]):
        semester_features[0, sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
        semester_masks[0, sem_idx] = 1.0
        
        courses = sem_data['courses'][:15]
        for course_idx, course in enumerate(courses):
            course_features[0, sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
            course_masks[0, sem_idx, course_idx] = 1.0
    
    # Dự đoán
    with torch.no_grad():
        prediction = model(course_features.to(device), 
                         semester_features.to(device), 
                         course_masks.to(device), 
                         semester_masks.to(device))
        predicted_gpa_norm = prediction[0, 0].item()
        predicted_cpa_norm = prediction[0, 1].item()
    
    # Chuyển đổi dự đoán về giá trị gốc
    predicted_gpa_original, predicted_cpa_original = inverse_normalize_gpa_cpa(
        [predicted_gpa_norm, predicted_cpa_norm], 
        original_gpa_col, 
        original_cpa_col
    )
    
    return {
        'student_id': student_id,
        'total_semesters': total_semesters,
        'input_semesters': len(input_semesters),
        'predicted_gpa': predicted_gpa_original,
        'predicted_cpa': predicted_cpa_original,
        'actual_gpa': actual_gpa_original,
        'actual_cpa': actual_cpa_original,
        'gpa_error': abs(predicted_gpa_original - actual_gpa_original),
        'cpa_error': abs(predicted_cpa_original - actual_cpa_original),
        'input_data': input_semesters,
        'target_data': target_semester
    }

def test_10_students():
    print("=== TEST DỰ ĐOÁN 10 SINH VIÊN CỤ THỂ ===")
    print("Sử dụng n-1 kỳ để dự đoán kỳ thứ n\n")
    
    print("Đang tải dữ liệu...")
    course_df = pd.read_csv('csv/ET1_K62_K63_K64.csv')
    perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
    
    print("Tìm 10 sinh viên có nhiều kỳ học nhất...")
    target_students = find_students_with_multiple_semesters(perf_df, min_semesters=3)
    
    print(f"Chọn được {len(target_students)} sinh viên:")
    for i, student_id in enumerate(target_students):
        semester_count = len(perf_df[perf_df['student_id'] == student_id])
        print(f"  {i+1}. Sinh viên {student_id}: {semester_count} kỳ")
    
    print("\nĐang load model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        model = StudentPerformancePredictor()
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        print("✓ Load model thành công")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return
    
    processor = StudentPerformanceDataProcessor()
    
    print("\n" + "="*80)
    print("CHI TIẾT DỰ ĐOÁN CHO TỪNG SINH VIÊN")
    print("="*80)
    
    results = []
    
    for i, student_id in enumerate(target_students):
        print(f"\n--- SINH VIÊN {i+1}: ID {student_id} ---")
        
        student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
        student_courses = course_df[course_df['student_id'] == student_id]
        
        print("Lịch sử học tập:")
        for idx, (_, row) in enumerate(student_perf.iterrows()):
            semester = row['Semester']
            gpa = safe_float_convert(row['GPA'])
            cpa = safe_float_convert(row['CPA'])
            tc_qua = safe_float_convert(row['TC qua'])
            warning = str(row['Warning']) if pd.notna(row['Warning']) else 'N/A'
            
            semester_courses = student_courses[student_courses['Semester'] == semester]
            course_count = len(semester_courses)
            
            marker = "→" if idx == len(student_perf) - 1 else " "
            print(f"  {marker} Kỳ {int(row['Relative Term']):2d}: GPA={gpa:.2f} | CPA={cpa:.2f} | "
                  f"TC={tc_qua:.0f} | Môn={course_count} | {warning}")
        
        try:
            result = predict_final_semester(student_id, course_df, perf_df, processor, model, device)
            
            if result:
                print(f"\nKết quả dự đoán:")
                print(f"  Input: {result['input_semesters']} kỳ đầu")
                print(f"  Dự đoán kỳ {result['total_semesters']}:")
                print(f"    GPA: {result['predicted_gpa']:.3f} (thực tế: {result['actual_gpa']:.3f}) - Sai số: {result['gpa_error']:.3f}")
                print(f"    CPA: {result['predicted_cpa']:.3f} (thực tế: {result['actual_cpa']:.3f}) - Sai số: {result['cpa_error']:.3f}")
                
                gpa_accuracy = "Rất tốt" if result['gpa_error'] < 0.2 else "Tốt" if result['gpa_error'] < 0.5 else "Cần cải thiện"
                cpa_accuracy = "Rất tốt" if result['cpa_error'] < 0.2 else "Tốt" if result['cpa_error'] < 0.5 else "Cần cải thiện"
                
                print(f"  Đánh giá: GPA - {gpa_accuracy}, CPA - {cpa_accuracy}")
                
                results.append(result)
            else:
                print("  ✗ Không thể dự đoán cho sinh viên này")
                
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
    
    # Tổng kết kết quả
    if results:
        print("\n" + "="*80)
        print("TỔNG KẾT KẾT QUẢ")
        print("="*80)
        
        avg_gpa_error = np.mean([r['gpa_error'] for r in results])
        avg_cpa_error = np.mean([r['cpa_error'] for r in results])
        
        print(f"Số sinh viên test thành công: {len(results)}/{len(target_students)}")
        print(f"Sai số trung bình:")
        print(f"  GPA: {avg_gpa_error:.3f}")
        print(f"  CPA: {avg_cpa_error:.3f}")
        
        # Phân loại độ chính xác (điều chỉnh threshold cho giá trị thực)
        gpa_excellent = sum(1 for r in results if r['gpa_error'] < 0.2)
        gpa_good = sum(1 for r in results if 0.2 <= r['gpa_error'] < 0.5)
        gpa_poor = sum(1 for r in results if r['gpa_error'] >= 0.5)
        
        cpa_excellent = sum(1 for r in results if r['cpa_error'] < 0.2)
        cpa_good = sum(1 for r in results if 0.2 <= r['cpa_error'] < 0.5)
        cpa_poor = sum(1 for r in results if r['cpa_error'] >= 0.5)
        
        print(f"\nPhân bố độ chính xác GPA:")
        print(f"  Rất tốt (< 0.2): {gpa_excellent}/{len(results)} ({gpa_excellent/len(results)*100:.1f}%)")
        print(f"  Tốt (0.2-0.5): {gpa_good}/{len(results)} ({gpa_good/len(results)*100:.1f}%)")
        print(f"  Cần cải thiện (≥ 0.5): {gpa_poor}/{len(results)} ({gpa_poor/len(results)*100:.1f}%)")
        
        print(f"\nPhân bố độ chính xác CPA:")
        print(f"  Rất tốt (< 0.2): {cpa_excellent}/{len(results)} ({cpa_excellent/len(results)*100:.1f}%)")
        print(f"  Tốt (0.2-0.5): {cpa_good}/{len(results)} ({cpa_good/len(results)*100:.1f}%)")
        print(f"  Cần cải thiện (≥ 0.5): {cpa_poor}/{len(results)} ({cpa_poor/len(results)*100:.1f}%)")
        
        # Sinh viên tốt nhất và kém nhất
        best_gpa = min(results, key=lambda x: x['gpa_error'])
        best_cpa = min(results, key=lambda x: x['cpa_error'])
        worst_gpa = max(results, key=lambda x: x['gpa_error'])
        worst_cpa = max(results, key=lambda x: x['cpa_error'])
        
        print(f"\nSinh viên có dự đoán tốt nhất:")
        print(f"  GPA: Sinh viên {best_gpa['student_id']} (sai số: {best_gpa['gpa_error']:.3f})")
        print(f"  CPA: Sinh viên {best_cpa['student_id']} (sai số: {best_cpa['cpa_error']:.3f})")
        
        print(f"\nSinh viên có dự đoán kém nhất:")
        print(f"  GPA: Sinh viên {worst_gpa['student_id']} (sai số: {worst_gpa['gpa_error']:.3f})")
        print(f"  CPA: Sinh viên {worst_cpa['student_id']} (sai số: {worst_cpa['cpa_error']:.3f})")

if __name__ == "__main__":
    test_10_students()