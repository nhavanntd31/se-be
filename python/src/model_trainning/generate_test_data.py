import pandas as pd
import os

def generate_test_data():
    print("=== GENERATING TEST DATA ===")
    
    course_file = "csv/ET1_K62_K63_K64.csv"
    performance_file = "csv/ET1_K62_K63_K64_performance.csv"
    
    if not os.path.exists(course_file):
        print(f"Error: Course file not found: {course_file}")
        return
    
    if not os.path.exists(performance_file):
        print(f"Error: Performance file not found: {performance_file}")
        return
    
    print("Loading original data...")
    course_df = pd.read_csv(course_file)
    performance_df = pd.read_csv(performance_file)
    
    print(f"Original data loaded:")
    print(f"  - Course records: {len(course_df)}")
    print(f"  - Performance records: {len(performance_df)}")
    print(f"  - Unique students: {len(course_df['student_id'].unique())}")
    
    print("\nFiltering students 1-10...")
    test_students = list(range(1, 11))
    
    course_filtered = course_df[course_df['student_id'].isin(test_students)].copy()
    performance_filtered = performance_df[performance_df['student_id'].isin(test_students)].copy()
    
    print(f"After filtering students 1-10:")
    print(f"  - Course records: {len(course_filtered)}")
    print(f"  - Performance records: {len(performance_filtered)}")
    print(f"  - Students found: {sorted(course_filtered['student_id'].unique())}")
    
    test_course_data = []
    test_performance_data = []
    
    print("\nProcessing each student...")
    for student_id in test_students:
        student_courses = course_filtered[course_filtered['student_id'] == student_id]
        student_performance = performance_filtered[performance_filtered['student_id'] == student_id]
        
        if len(student_courses) == 0 or len(student_performance) == 0:
            print(f"  Student {student_id}: No data found, skipping")
            continue
        
        student_semesters_course = student_courses['Semester'].unique()
        student_semesters_perf = student_performance['Semester'].unique()
        
        all_semesters = sorted(set(list(student_semesters_course) + list(student_semesters_perf)))
        
        if len(all_semesters) < 2:
            print(f"  Student {student_id}: Only {len(all_semesters)} semester(s), skipping")
            continue
        
        last_semester = all_semesters[-1]
        
        print(f"  Student {student_id}:")
        print(f"    - Total semesters: {len(all_semesters)}")
        print(f"    - Last semester: {last_semester}")
        print(f"    - Course records: {len(student_courses)}")
        print(f"    - Performance records: {len(student_performance)}")
        
        courses_without_last = student_courses[student_courses['Semester'] != last_semester]
        performance_without_last = student_performance[student_performance['Semester'] != last_semester]
        
        print(f"    - After removing last semester:")
        print(f"      - Course records: {len(courses_without_last)}")
        print(f"      - Performance records: {len(performance_without_last)}")
        
        if len(courses_without_last) > 0:
            test_course_data.append(courses_without_last)
        
        if len(performance_without_last) > 0:
            test_performance_data.append(performance_without_last)
    
    if not test_course_data or not test_performance_data:
        print("Error: No valid test data generated")
        return
    
    print("\nCombining test data...")
    final_course_df = pd.concat(test_course_data, ignore_index=True)
    final_performance_df = pd.concat(test_performance_data, ignore_index=True)
    
    print(f"Final test data:")
    print(f"  - Course records: {len(final_course_df)}")
    print(f"  - Performance records: {len(final_performance_df)}")
    print(f"  - Students in final dataset: {sorted(final_course_df['student_id'].unique())}")
    
    output_course_file = "csv/test_course_data.csv"
    output_performance_file = "csv/test_performance_data.csv"
    
    print(f"\nSaving test data...")
    final_course_df.to_csv(output_course_file, index=False)
    final_performance_df.to_csv(output_performance_file, index=False)
    
    print(f"Test data saved:")
    print(f"  - Course data: {output_course_file}")
    print(f"  - Performance data: {output_performance_file}")
    
    print("\n=== SUMMARY ===")
    for student_id in sorted(final_course_df['student_id'].unique()):
        student_courses = final_course_df[final_course_df['student_id'] == student_id]
        student_perf = final_performance_df[final_performance_df['student_id'] == student_id]
        
        semesters_course = sorted(student_courses['Semester'].unique())
        semesters_perf = sorted(student_perf['Semester'].unique())
        
        print(f"Student {student_id}:")
        print(f"  - Course semesters: {semesters_course}")
        print(f"  - Performance semesters: {semesters_perf}")
        print(f"  - Total course records: {len(student_courses)}")
        print(f"  - Total performance records: {len(student_perf)}")
        
        if len(student_perf) > 0:
            latest_perf = student_perf.iloc[-1]
            print(f"  - Latest GPA: {latest_perf['GPA']}")
            print(f"  - Latest CPA: {latest_perf['CPA']}")
    
    print(f"\n✅ Test data generation completed!")
    print(f"Files created:")
    print(f"  - {output_course_file}")
    print(f"  - {output_performance_file}")

if __name__ == "__main__":
    generate_test_data() 