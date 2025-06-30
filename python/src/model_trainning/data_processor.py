import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple, Optional
import warnings
import pickle
warnings.filterwarnings('ignore')

class StudentPerformanceDataProcessor:
    def __init__(self):
        self.grade_mapping = {
            'A+': 4.0, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5, 'C': 2,
            'D+': 1.5, 'D': 1, 'F': 0, 'X': 0, 'W': 0
        }
        self.warning_mapping = {
            'Mức 0': 0, 'Mức 1': 1, 'Mức 2': 2, 'Mức 3': 3
        }
        self.course_scaler = StandardScaler()
        self.performance_scaler = StandardScaler()
        
    def clean_numeric_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            df[col] = df[col].replace([np.inf, -np.inf], mean_val)
        return df
        
    def preprocess_course_data(self, course_df: pd.DataFrame) -> pd.DataFrame:
        df = course_df.copy()
        
        df['Final Grade Numeric'] = df['Final Grade'].map(self.grade_mapping)
        
        numeric_features = ['Continuous Assessment Score', 'Exam Score', 'Credits', 
                          'Final Grade Numeric']
        
        df = self.clean_numeric_data(df, numeric_features)
        
        df['Course_Category'] = df['Course ID'].str[:2]
        course_cat_encoder = LabelEncoder()
        df['Course_Category_Encoded'] = course_cat_encoder.fit_transform(df['Course_Category'])
        
        df['Pass_Status'] = (df['Final Grade Numeric'] >= 1.0).astype(int)
        df['Grade_Points'] = df['Final Grade Numeric'] * df['Credits']
        
        all_features = numeric_features + ['Course_Category_Encoded', 'Pass_Status', 'Grade_Points']
        
        df[all_features] = self.course_scaler.fit_transform(df[all_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + all_features]
    
    def preprocess_performance_data(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        df = perf_df.copy()
        
        numeric_cols = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg']
        df = self.clean_numeric_data(df, numeric_cols)
        
        df['Warning_Numeric'] = df['Warning'].map(self.warning_mapping).fillna(0)
        
        df['Level_Year'] = df['Level'].str.extract('(\d+)').astype(float).fillna(1)
        
        df['Pass_Rate'] = df['TC qua'] / (df['Reg'] + 1e-8)
        df['Debt_Rate'] = df['Debt'] / (df['Reg'] + 1e-8)
        df['Accumulation_Rate'] = df['Acc'] / (df['Relative Term'] * 20 + 1e-8)
        
        rate_cols = ['Pass_Rate', 'Debt_Rate', 'Accumulation_Rate']
        df = self.clean_numeric_data(df, rate_cols)
        
        performance_features = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                              'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                              'Debt_Rate', 'Accumulation_Rate']
        
        df[performance_features] = self.performance_scaler.fit_transform(df[performance_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + performance_features]
    
    def save_scalers(self, course_scaler_path='course_scaler.pkl', performance_scaler_path='feature_scaler.pkl'):
        with open(course_scaler_path, 'wb') as f:
            pickle.dump(self.course_scaler, f)
        with open(performance_scaler_path, 'wb') as f:
            pickle.dump(self.performance_scaler, f)
        print(f"Scalers saved: {course_scaler_path}, {performance_scaler_path}")
    
    def load_scalers(self, course_scaler_path='course_scaler.pkl', performance_scaler_path='feature_scaler.pkl'):
        try:
            with open(course_scaler_path, 'rb') as f:
                self.course_scaler = pickle.load(f)
            with open(performance_scaler_path, 'rb') as f:
                self.performance_scaler = pickle.load(f)
            print(f"Scalers loaded: {course_scaler_path}, {performance_scaler_path}")
            return True
        except Exception as e:
            print(f"Error loading scalers: {e}")
            return False
    
    def denormalize_performance(self, normalized_data):
        dummy_data = np.zeros((len(normalized_data), 11))
        dummy_data[:, :normalized_data.shape[1]] = normalized_data
        
        denormalized = self.performance_scaler.inverse_transform(dummy_data)
        return denormalized[:, :normalized_data.shape[1]]

def is_active_semester(perf_row, course_count):
    gpa = perf_row['GPA'] if 'GPA' in perf_row else 0
    tc_qua = perf_row['TC qua'] if 'TC qua' in perf_row else 0
    
    if pd.isna(gpa) or gpa == 0:
        return False
    if course_count == 0:
        return False
    if pd.isna(tc_qua) or tc_qua == 0:
        return False
    
    return True

def create_student_sequences(course_df: pd.DataFrame, perf_df: pd.DataFrame, 
                           processor: StudentPerformanceDataProcessor) -> List[Dict]:
    course_processed = processor.preprocess_course_data(course_df)
    perf_processed = processor.preprocess_performance_data(perf_df)
    sequences = []
    skipped_students = []
    dropped_out_students = []
    
    for student_id in course_processed['student_id'].unique():
        student_courses = course_processed[course_processed['student_id'] == student_id]
        student_perf = perf_processed[perf_processed['student_id'] == student_id]
        
        if len(student_perf) < 2:
            skipped_students.append(f"{student_id} (ít hơn 2 kỳ)")
            continue
        
        semester_data = []
        has_null = False
        
        for _, perf_row in student_perf.iterrows():
            semester = perf_row['Semester']
            semester_courses = student_courses[student_courses['Semester'] == semester]
            
            if not is_active_semester(perf_row, len(semester_courses)):
                print(f"Sinh viên {student_id} đã bỏ học/nghỉ học từ kỳ {semester}")
                break
            
            if len(semester_courses) == 0:
                continue
                
            course_features = semester_courses[['Continuous Assessment Score', 'Exam Score', 'Credits',
                                             'Final Grade Numeric', 'Course_Category_Encoded', 
                                             'Pass_Status', 'Grade_Points']].values.tolist()
            
            performance_features = perf_row[['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                                           'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                                           'Debt_Rate', 'Accumulation_Rate']].values.tolist()
            
            if np.isnan(course_features).any() or np.isnan(performance_features).any():
                has_null = True
                break
                
            semester_data.append({
                'semester': semester,
                'relative_term': perf_row['Relative Term'],
                'courses': course_features,
                'performance': performance_features,
                'gpa': perf_row['GPA'] if 'GPA' in perf_row else 0,
                'tc_qua': perf_row['TC qua'] if 'TC qua' in perf_row else 0
            })
        
        if has_null:
            skipped_students.append(f"{student_id} (dữ liệu null)")
            continue
            
        if len(semester_data) < 2:
            dropped_out_students.append(f"{student_id} (bỏ học sớm)")
            continue
            
        semester_data.sort(key=lambda x: x['relative_term'])
        
        input_semesters = semester_data[:4]
        target_semester = semester_data[4]
        
        target_gpa = target_semester['gpa']
        target_tc = target_semester['tc_qua']
        
        if target_gpa > 0 and target_tc > 0:
            sequences.append({
                'student_id': student_id,
                'semesters': input_semesters,
                'target_gpa': target_semester['performance'][0],
                'target_cpa': target_semester['performance'][1]
            })
        else:
            dropped_out_students.append(f"{student_id} (không có sequence hợp lệ)")
    
    print(f"\nSố sinh viên bị loại do dữ liệu null/thiếu: {len(skipped_students)}")
    print(f"Số sinh viên bỏ học/nghỉ học: {len(dropped_out_students)}")
    print(f"Danh sách sinh viên bỏ học: {dropped_out_students[:10]}...")
    
    return sequences

def create_temporal_train_test_sequences(course_df: pd.DataFrame, perf_df: pd.DataFrame, 
                                       processor: StudentPerformanceDataProcessor, 
                                       train_ratio: float = 0.6) -> Tuple[List[Dict], List[Dict]]:
    course_processed = processor.preprocess_course_data(course_df)
    perf_processed = processor.preprocess_performance_data(perf_df)
    
    all_students = course_processed['student_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_students)
    
    n_train_students = int(len(all_students) * train_ratio)
    train_students = set(np.random.choice(all_students, size=n_train_students, replace=False))
    test_students = set(np.setdiff1d(all_students, list(train_students)))
    
    train_sequences = []
    test_sequences = []
    train_skipped = []
    test_skipped = []
    
    print(f"Total students: {len(all_students)}")
    print(f"Train students: {len(train_students)}")
    print(f"Test students: {len(test_students)}")
    
    for student_id in all_students:
        student_courses = course_processed[course_processed['student_id'] == student_id]
        student_perf = perf_processed[perf_processed['student_id'] == student_id]
        
        if len(student_perf) < 5:
            if student_id in train_students:
                train_skipped.append(f"{student_id} (ít hơn 5 kỳ)")
            else:
                test_skipped.append(f"{student_id} (ít hơn 5 kỳ)")
            continue
        
        semester_data = []
        has_null = False
        
        for _, perf_row in student_perf.iterrows():
            semester = perf_row['Semester']
            semester_courses = student_courses[student_courses['Semester'] == semester]
            
            if not is_active_semester(perf_row, len(semester_courses)):
                break
            
            if len(semester_courses) == 0:
                continue
                
            course_features = semester_courses[['Continuous Assessment Score', 'Exam Score', 'Credits',
                                             'Final Grade Numeric', 'Course_Category_Encoded', 
                                             'Pass_Status', 'Grade_Points']].values.tolist()
            
            performance_features = perf_row[['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                                           'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                                           'Debt_Rate', 'Accumulation_Rate']].values.tolist()
            
            if np.isnan(course_features).any() or np.isnan(performance_features).any():
                has_null = True
                break
                
            semester_data.append({
                'semester': semester,
                'relative_term': perf_row['Relative Term'],
                'courses': course_features,
                'performance': performance_features,
                'gpa': perf_row['GPA'] if 'GPA' in perf_row else 0,
                'tc_qua': perf_row['TC qua'] if 'TC qua' in perf_row else 0
            })
        
        if has_null or len(semester_data) < 5:
            if student_id in train_students:
                train_skipped.append(f"{student_id} (dữ liệu thiếu)")
            else:
                test_skipped.append(f"{student_id} (dữ liệu thiếu)")
            continue
            
        semester_data.sort(key=lambda x: x['relative_term'])
        
        if student_id in train_students:
            if len(semester_data) >= 5:
                target_semester = semester_data[4]
                if target_semester['gpa'] > 0 and target_semester['tc_qua'] > 0:
                    train_sequences.append({
                        'student_id': student_id,
                        'semesters': semester_data[:4],
                        'target_gpa': target_semester['performance'][0],
                        'target_cpa': target_semester['performance'][1],
                        'sequence_type': 'kỳ_1-4_dự_đoán_kỳ_5'
                    })
        else:
            for start_idx in range(len(semester_data) - 4):
                target_idx = start_idx + 4
                if target_idx < len(semester_data):
                    target_semester = semester_data[target_idx]
                    if target_semester['gpa'] > 0 and target_semester['tc_qua'] > 0:
                        sequence_type = f"kỳ_{start_idx+1}-{start_idx+4}_dự_đoán_kỳ_{target_idx+1}"
                        test_sequences.append({
                            'student_id': student_id,
                            'semesters': semester_data[start_idx:start_idx+4],
                            'target_gpa': target_semester['performance'][0],
                            'target_cpa': target_semester['performance'][1],
                            'sequence_type': sequence_type
                        })
    
    print(f"\nTrain sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Train skipped: {len(train_skipped)}")
    print(f"Test skipped: {len(test_skipped)}")
    
    test_sequence_types = {}
    for seq in test_sequences:
        seq_type = seq['sequence_type']
        test_sequence_types[seq_type] = test_sequence_types.get(seq_type, 0) + 1
    
    print(f"\nTest sequence distribution:")
    for seq_type, count in sorted(test_sequence_types.items()):
        print(f"  {seq_type}: {count} sequences")
    
    return train_sequences, test_sequences

def compute_denormalized_metrics(predictions_normalized, targets_normalized, processor):
    """
    Tính metrics trên dữ liệu denormalized (scale gốc)
    """
    try:
        dummy_targets = np.zeros((len(targets_normalized), 11))
        dummy_targets[:, 0] = targets_normalized[:, 0] 
        dummy_targets[:, 1] = targets_normalized[:, 1]
        
        dummy_predictions = np.zeros((len(predictions_normalized), 11))
        dummy_predictions[:, 0] = predictions_normalized[:, 0]
        dummy_predictions[:, 1] = predictions_normalized[:, 1]
        # for i in range(len(dummy_targets)):
        #     print(f"Index {i}:")
        #     print(f"  Dummy targets: {dummy_targets[i]}")
        #     print(f"  Dummy predictions: {dummy_predictions[i]}")
        targets_denorm = processor.performance_scaler.inverse_transform(dummy_targets)
        predictions_denorm = processor.performance_scaler.inverse_transform(dummy_predictions)
        
        gpa_mse = mean_squared_error(targets_denorm[:, 0], predictions_denorm[:, 0])
        cpa_mse = mean_squared_error(targets_denorm[:, 1], predictions_denorm[:, 1])
        gpa_mae = mean_absolute_error(targets_denorm[:, 0], predictions_denorm[:, 0])
        cpa_mae = mean_absolute_error(targets_denorm[:, 1], predictions_denorm[:, 1])
        gpa_r2 = r2_score(targets_denorm[:, 0], predictions_denorm[:, 0])
        cpa_r2 = r2_score(targets_denorm[:, 1], predictions_denorm[:, 1])
        
        return {
            'gpa_mse_denorm': gpa_mse,
            'cpa_mse_denorm': cpa_mse,
            'gpa_mae_denorm': gpa_mae,
            'cpa_mae_denorm': cpa_mae,
            'gpa_r2_denorm': gpa_r2,
            'cpa_r2_denorm': cpa_r2,
            'predictions_denorm': predictions_denorm,
            'targets_denorm': targets_denorm
        }
    except Exception as e:
        print(f"Warning: Could not compute denormalized metrics: {e}")
        return {
            'gpa_mse_denorm': float('inf'),
            'cpa_mse_denorm': float('inf'),
            'gpa_mae_denorm': float('inf'),
            'cpa_mae_denorm': float('inf'),
            'gpa_r2_denorm': float('-inf'),
            'cpa_r2_denorm': float('-inf')
        }

def load_and_prepare_data(course_csv_path, performance_csv_path):
    """
    Load và chuẩn bị dữ liệu - phiên bản cũ (sử dụng random split)
    """
    print(f"Loading data...")
    print(f"Course CSV: {course_csv_path}")
    print(f"Performance CSV: {performance_csv_path}")
    
    course_df = pd.read_csv(course_csv_path)
    perf_df = pd.read_csv(performance_csv_path)
    
    print(f"Course data: {len(course_df)} records")
    print(f"Performance data: {len(perf_df)} records")
    
    print(f"\nProcessing data...")
    processor = StudentPerformanceDataProcessor()
    sequences = create_student_sequences(course_df, perf_df, processor)
    
    print(f"Created {len(sequences)} training sequences")
    
    return sequences, processor

def load_and_prepare_temporal_data(course_csv_path, performance_csv_path, train_ratio=0.6):
    """
    Load và chuẩn bị dữ liệu - phiên bản temporal split (tránh data leak)
    """
    print(f"Loading data for temporal train/test split...")
    print(f"Course CSV: {course_csv_path}")
    print(f"Performance CSV: {performance_csv_path}")
    print(f"Train ratio: {train_ratio}")
    
    course_df = pd.read_csv(course_csv_path)
    perf_df = pd.read_csv(performance_csv_path)
    
    print(f"Course data: {len(course_df)} records")
    print(f"Performance data: {len(perf_df)} records")
    
    print(f"\nProcessing data with temporal split...")
    processor = StudentPerformanceDataProcessor()
    train_sequences, test_sequences = create_temporal_train_test_sequences(
        course_df, perf_df, processor, train_ratio
    )
    
    return train_sequences, test_sequences, processor 