import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def analyze_course_data(course_df: pd.DataFrame) -> Dict:
    print("=== PHÂN TÍCH DỮ LIỆU MÔhọc ===")
    
    print(f"📊 Tổng số records: {len(course_df):,}")
    print(f"👥 Số sinh viên unique: {course_df['student_id'].nunique():,}")
    print(f"📚 Số môn học unique: {course_df['Course ID'].nunique():,}")
    print(f"📅 Số semester unique: {course_df['Semester'].nunique():,}")
    
    print(f"\n📊 Phân bố điểm Final Grade:")
    grade_dist = course_df['Final Grade'].value_counts().sort_index()
    for grade, count in grade_dist.items():
        percentage = (count / len(course_df)) * 100
        print(f"  {grade}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📊 Thống kê điểm số:")
    print(f"  Điểm quá trình - Trung bình: {course_df['Continuous Assessment Score'].mean():.2f}")
    print(f"  Điểm thi - Trung bình: {course_df['Exam Score'].mean():.2f}")
    
    print(f"\n📚 Top 10 môn học phổ biến:")
    top_courses = course_df['Course ID'].value_counts().head(10)
    for course, count in top_courses.items():
        print(f"  {course}: {count:,} lượt đăng ký")
    
    print(f"\n📅 Phân bố theo Relative Term:")
    term_dist = course_df['Relative Term'].value_counts().sort_index()
    for term, count in term_dist.items():
        print(f"  Kỳ {term}: {count:,} records")
    
    return {
        'total_records': len(course_df),
        'unique_students': course_df['student_id'].nunique(),
        'unique_courses': course_df['Course ID'].nunique(),
        'grade_distribution': grade_dist.to_dict(),
        'avg_continuous_score': course_df['Continuous Assessment Score'].mean(),
        'avg_exam_score': course_df['Exam Score'].mean()
    }

def analyze_performance_data(perf_df: pd.DataFrame) -> Dict:
    print("\n=== PHÂN TÍCH DỮ LIỆU PERFORMANCE ===")
    
    print(f"📊 Tổng số records: {len(perf_df):,}")
    print(f"👥 Số sinh viên unique: {perf_df['student_id'].nunique():,}")
    
    gpa_stats = perf_df['GPA'].describe()
    cpa_stats = perf_df['CPA'].describe()
    
    print(f"\n📈 Thống kê GPA:")
    print(f"  Trung bình: {gpa_stats['mean']:.3f}")
    print(f"  Độ lệch chuẩn: {gpa_stats['std']:.3f}")
    print(f"  Min: {gpa_stats['min']:.3f}")
    print(f"  Max: {gpa_stats['max']:.3f}")
    
    print(f"\n📈 Thống kê CPA:")
    print(f"  Trung bình: {cpa_stats['mean']:.3f}")
    print(f"  Độ lệch chuẩn: {cpa_stats['std']:.3f}")
    print(f"  Min: {cpa_stats['min']:.3f}")
    print(f"  Max: {cpa_stats['max']:.3f}")
    
    print(f"\n⚠️ Phân bố Warning levels:")
    warning_dist = perf_df['Warning'].value_counts()
    for warning, count in warning_dist.items():
        percentage = (count / len(perf_df)) * 100
        print(f"  {warning}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🎓 Phân bố Level (năm học):")
    level_dist = perf_df['Level'].value_counts()
    for level, count in level_dist.items():
        percentage = (count / len(perf_df)) * 100
        print(f"  {level}: {count:,} ({percentage:.1f}%)")
    
    return {
        'gpa_mean': gpa_stats['mean'],
        'gpa_std': gpa_stats['std'],
        'cpa_mean': cpa_stats['mean'],
        'cpa_std': cpa_stats['std'],
        'warning_distribution': warning_dist.to_dict()
    }

def analyze_student_trajectories(course_df: pd.DataFrame, perf_df: pd.DataFrame):
    print("\n=== PHÂN TÍCH QUÁ TRÌNH HỌC TẬP SINH VIÊN ===")
    
    students_with_multiple_terms = perf_df.groupby('student_id').size()
    students_with_enough_data = students_with_multiple_terms[students_with_multiple_terms >= 3]
    
    print(f"👥 Sinh viên có ít nhất 3 kỳ học: {len(students_with_enough_data):,}")
    print(f"📊 Trung bình số kỳ/sinh viên: {students_with_multiple_terms.mean():.1f}")
    print(f"📊 Max số kỳ: {students_with_multiple_terms.max()}")
    
    sample_students = students_with_enough_data.head(5).index
    
    print(f"\n📈 Phân tích xu hướng GPA của 5 sinh viên mẫu:")
    for student_id in sample_students:
        student_perf = perf_df[perf_df['student_id'] == student_id].sort_values('Relative Term')
        gpa_values = student_perf['GPA'].dropna()
        
        if len(gpa_values) >= 2:
            gpa_trend = gpa_values.iloc[-1] - gpa_values.iloc[0]
            print(f"  Sinh viên {student_id}: {gpa_values.iloc[0]:.2f} → {gpa_values.iloc[-1]:.2f} "
                  f"({gpa_trend:+.2f})")

def create_visualizations(course_df: pd.DataFrame, perf_df: pd.DataFrame):
    print("\n=== TẠO BIỂU ĐỒ PHÂN TÍCH ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    grade_mapping = {'A+': 4.0, 'A': 3.7, 'B+': 3.3, 'B': 3.0, 'C+': 2.3, 'C': 2.0,
                     'D+': 1.3, 'D': 1.0, 'F': 0.0, 'X': 0.0, 'W': 0.0}
    course_df['Grade_Numeric'] = course_df['Final Grade'].map(grade_mapping)
    
    grade_counts = course_df['Final Grade'].value_counts()
    axes[0,0].pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Phân bố Final Grade')
    
    axes[0,1].hist(perf_df['GPA'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Phân bố GPA')
    axes[0,1].set_xlabel('GPA')
    axes[0,1].set_ylabel('Tần suất')
    
    axes[0,2].hist(perf_df['CPA'].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0,2].set_title('Phân bố CPA')
    axes[0,2].set_xlabel('CPA')
    axes[0,2].set_ylabel('Tần suất')
    
    term_counts = course_df['Relative Term'].value_counts().sort_index()
    axes[1,0].bar(term_counts.index, term_counts.values)
    axes[1,0].set_title('Số lượng records theo Relative Term')
    axes[1,0].set_xlabel('Relative Term')
    axes[1,0].set_ylabel('Số lượng records')
    
    warning_counts = perf_df['Warning'].value_counts()
    axes[1,1].bar(range(len(warning_counts)), warning_counts.values)
    axes[1,1].set_xticks(range(len(warning_counts)))
    axes[1,1].set_xticklabels(warning_counts.index, rotation=45)
    axes[1,1].set_title('Phân bố Warning Level')
    axes[1,1].set_ylabel('Số lượng')
    
    students_with_multiple_terms = perf_df.groupby('student_id').size()
    axes[1,2].hist(students_with_multiple_terms.values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,2].set_title('Phân bố số kỳ học/sinh viên')
    axes[1,2].set_xlabel('Số kỳ học')
    axes[1,2].set_ylabel('Số sinh viên')
    
    plt.tight_layout()
    plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Đã lưu biểu đồ phân tích vào 'data_analysis_plots.png'")

def main():
    print("🔍 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU SINH VIÊN")
    print("=" * 60)
    
    try:
        course_df = pd.read_csv('csv/ET1_K62_K63_K64.csv')
        perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
        
        course_stats = analyze_course_data(course_df)
        perf_stats = analyze_performance_data(perf_df)
        analyze_student_trajectories(course_df, perf_df)
        
        create_visualizations(course_df, perf_df)
        
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH PHÂN TÍCH DỮ LIỆU")
        
        return {
            'course_stats': course_stats,
            'performance_stats': perf_stats
        }
        
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file dữ liệu: {e}")
        print("Vui lòng đảm bảo các file CSV đã được đặt trong thư mục 'csv/'")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi phân tích dữ liệu: {e}")
        return None

if __name__ == "__main__":
    main() 