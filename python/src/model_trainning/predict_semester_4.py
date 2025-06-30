import pandas as pd
import numpy as np
import torch
import random
from student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    StudentPerformancePredictor,
)

def safe_float_convert(value):
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def inverse_normalize_gpa_cpa(normalized_values, original_gpa_col, original_cpa_col):
    gpa_mean = original_gpa_col.mean()
    gpa_std = original_gpa_col.std()
    cpa_mean = original_cpa_col.mean()
    cpa_std = original_cpa_col.std()
    
    original_gpa = normalized_values[0] * gpa_std + gpa_mean
    original_cpa = normalized_values[1] * cpa_std + cpa_mean
    
    return original_gpa, original_cpa

def find_students_with_at_least_4_semesters(perf_df):
    student_semester_counts = perf_df.groupby('student_id')['Relative Term'].count()
    qualified_students = student_semester_counts[student_semester_counts >= 4]
    return qualified_students.index.tolist()

def predict_semester_4_from_first_3(student_id, course_df, perf_df, processor, model, device):
    student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
    student_courses = course_df[course_df['student_id'] == student_id]
    
    if len(student_perf) < 4:
        return None
    
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
    
    if len(semester_data) < 4:
        return None
    
    input_semesters = semester_data[:3]
    target_semester = semester_data[3]
    
    target_semester_info = student_perf.iloc[3]
    actual_gpa_original = safe_float_convert(target_semester_info['GPA'])
    actual_cpa_original = safe_float_convert(target_semester_info['CPA'])
    
    course_features = torch.zeros(1, 10, 15, 7)
    semester_features = torch.zeros(1, 10, 11)
    course_masks = torch.zeros(1, 10, 15)
    semester_masks = torch.zeros(1, 10)
    
    for sem_idx, sem_data in enumerate(input_semesters):
        semester_features[0, sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
        semester_masks[0, sem_idx] = 1.0
        
        courses = sem_data['courses'][:15]
        for course_idx, course in enumerate(courses):
            course_features[0, sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
            course_masks[0, sem_idx, course_idx] = 1.0
    
    with torch.no_grad():
        prediction = model(course_features.to(device), 
                         semester_features.to(device), 
                         course_masks.to(device), 
                         semester_masks.to(device))
        predicted_gpa_norm = prediction[0, 0].item()
        predicted_cpa_norm = prediction[0, 1].item()
    
    predicted_gpa_original, predicted_cpa_original = inverse_normalize_gpa_cpa(
        [predicted_gpa_norm, predicted_cpa_norm], 
        original_gpa_col, 
        original_cpa_col
    )
    
    return {
        'student_id': student_id,
        'predicted_gpa': predicted_gpa_original,
        'predicted_cpa': predicted_cpa_original,
        'actual_gpa': actual_gpa_original,
        'actual_cpa': actual_cpa_original,
        'gpa_error': abs(predicted_gpa_original - actual_gpa_original),
        'cpa_error': abs(predicted_cpa_original - actual_cpa_original),
        'input_semesters_data': input_semesters,
        'target_semester_data': target_semester
    }

def test_predict_semester_4():
    print("=== DỰ ĐOÁN KỲ 4 TỪ 3 KỲ ĐẦU CỦA 10 SINH VIÊN NGẪU NHIÊN ===")
    print("Sử dụng 3 kỳ đầu để dự đoán kỳ 4\n")
    
    print("Đang tải dữ liệu...")
    course_df = pd.read_csv('csv/ET1_K62_K63_K64.csv')
    perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
    
    print("Tìm sinh viên có ít nhất 4 kỳ học...")
    qualified_students = find_students_with_at_least_4_semesters(perf_df)
    print(f"Tìm được {len(qualified_students)} sinh viên có ít nhất 4 kỳ")
    
    print("Chọn ngẫu nhiên 10 sinh viên...")
    random.seed(42)
    selected_students = random.sample(qualified_students, min(10, len(qualified_students)))
    
    print(f"Danh sách 10 sinh viên được chọn:")
    for i, student_id in enumerate(selected_students):
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
    
    for i, student_id in enumerate(selected_students):
        print(f"\n--- SINH VIÊN {i+1}: ID {student_id} ---")
        
        student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
        student_courses = course_df[course_df['student_id'] == student_id]
        
        print("Lịch sử học tập (3 kỳ đầu + kỳ 4 thực tế):")
        for idx, (_, row) in enumerate(student_perf.head(4).iterrows()):
            semester = row['Semester']
            gpa = safe_float_convert(row['GPA'])
            cpa = safe_float_convert(row['CPA'])
            tc_qua = safe_float_convert(row['TC qua'])
            warning = str(row['Warning']) if pd.notna(row['Warning']) else 'N/A'
            
            semester_courses = student_courses[student_courses['Semester'] == semester]
            course_count = len(semester_courses)
            
            if idx < 3:
                marker = " "
                status = "(input)"
            else:
                marker = "→"
                status = "(target)"
            
            print(f"  {marker} Kỳ {int(row['Relative Term']):2d}: GPA={gpa:.2f} | CPA={cpa:.2f} | "
                  f"TC={tc_qua:.0f} | Môn={course_count} | {warning} {status}")
        
        try:
            result = predict_semester_4_from_first_3(student_id, course_df, perf_df, processor, model, device)
            
            if result:
                print(f"\nKết quả dự đoán kỳ 4:")
                print(f"    GPA: {result['predicted_gpa']:.3f} (thực tế: {result['actual_gpa']:.3f}) - Sai số: {result['gpa_error']:.3f}")
                print(f"    CPA: {result['predicted_cpa']:.3f} (thực tế: {result['actual_cpa']:.3f}) - Sai số: {result['cpa_error']:.3f}")
                
                gpa_accuracy = "Rất tốt" if result['gpa_error'] < 0.2 else "Tốt" if result['gpa_error'] < 0.5 else "Cần cải thiện"
                cpa_accuracy = "Rất tốt" if result['cpa_error'] < 0.2 else "Tốt" if result['cpa_error'] < 0.5 else "Cần cải thiện"
                
                print(f"  Đánh giá: GPA - {gpa_accuracy}, CPA - {cpa_accuracy}")
                
                gpa_percentage_error = (result['gpa_error'] / result['actual_gpa'] * 100) if result['actual_gpa'] > 0 else 0
                cpa_percentage_error = (result['cpa_error'] / result['actual_cpa'] * 100) if result['actual_cpa'] > 0 else 0
                print(f"  Sai số tương đối: GPA {gpa_percentage_error:.1f}%, CPA {cpa_percentage_error:.1f}%")
                
                results.append(result)
            else:
                print("  ✗ Không thể dự đoán cho sinh viên này")
                
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
    
    if results:
        print("\n" + "="*80)
        print("TỔNG KẾT KẾT QUẢ DỰ ĐOÁN KỲ 4")
        print("="*80)
        
        avg_gpa_error = np.mean([r['gpa_error'] for r in results])
        avg_cpa_error = np.mean([r['cpa_error'] for r in results])
        
        print(f"Số sinh viên test thành công: {len(results)}/{len(selected_students)}")
        print(f"Sai số trung bình:")
        print(f"  GPA: {avg_gpa_error:.3f}")
        print(f"  CPA: {avg_cpa_error:.3f}")
        
        avg_gpa_percentage = np.mean([(r['gpa_error'] / r['actual_gpa'] * 100) if r['actual_gpa'] > 0 else 0 for r in results])
        avg_cpa_percentage = np.mean([(r['cpa_error'] / r['actual_cpa'] * 100) if r['actual_cpa'] > 0 else 0 for r in results])
        print(f"Sai số tương đối trung bình:")
        print(f"  GPA: {avg_gpa_percentage:.1f}%")
        print(f"  CPA: {avg_cpa_percentage:.1f}%")
        
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
        
        print(f"\n=== KẾT LUẬN ===")
        if avg_gpa_error < 0.3 and avg_cpa_error < 0.3:
            print("✓ Model cho kết quả dự đoán TỐT cho kỳ 4")
        elif avg_gpa_error < 0.5 and avg_cpa_error < 0.5:
            print("~ Model cho kết quả dự đoán CHẤP NHẬN ĐƯỢC cho kỳ 4")
        else:
            print("✗ Model cần cải thiện cho việc dự đoán kỳ 4")

if __name__ == "__main__":
    test_predict_semester_4() 