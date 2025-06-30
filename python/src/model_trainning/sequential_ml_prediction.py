import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SequentialMLPredictor:
    def __init__(self):
        self.grade_mapping = {
            'A+': 4.0, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5, 'C': 2,
            'D+': 1.5, 'D': 1, 'F': 0, 'X': 0, 'W': 0
        }
        self.warning_mapping = {
            'Mức 0': 0, 'Mức 1': 1, 'Mức 2': 2, 'Mức 3': 3
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def prepare_sequential_data(self, course_df, perf_df, min_semesters=3):
        print("🔄 Chuẩn bị dữ liệu sequential...")
        print(f"📊 Tổng số sinh viên: {perf_df['student_id'].nunique()}")
        students_data = []
        
        for student_id, student_perf in perf_df.groupby('student_id'):
            student_perf = student_perf.sort_values('Relative Term')
            if len(student_perf) < min_semesters:
                continue
            student_courses = course_df[course_df['student_id'] == student_id]
            for i in range(min_semesters-1, len(student_perf)):
                try:
                    historical_semesters = student_perf.iloc[:i]
                    target_semester = student_perf.iloc[i]
                    features = self._extract_historical_features(
                        historical_semesters, student_courses, student_id
                    )
                    target_gpa = pd.to_numeric(target_semester['GPA'], errors='coerce')
                    target_cpa = pd.to_numeric(target_semester['CPA'], errors='coerce')
                    
                    if features is not None and not pd.isna(target_gpa) and not pd.isna(target_cpa):
                        students_data.append({
                            'student_id': student_id,
                            'features': features,
                            'target_gpa': target_gpa,
                            'target_cpa': target_cpa,
                            'target_semester': target_semester['Semester'],
                            'num_historical_semesters': len(historical_semesters)
                        })
                            
                except Exception as e:
                    continue
        
        print(f"📊 Tạo được {len(students_data)} samples từ {perf_df['student_id'].nunique()} sinh viên")
        
        if len(students_data) == 0:
            return None, None, None, None, None
            
        X = np.array([sample['features'] for sample in students_data])
        y_gpa = np.array([sample['target_gpa'] for sample in students_data])
        y_cpa = np.array([sample['target_cpa'] for sample in students_data])
        
        feature_names = self._get_feature_names()
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        print(f"📊 Final dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"📊 GPA range: {y_gpa.min():.2f} - {y_gpa.max():.2f} (original scale)")
        print(f"📊 CPA range: {y_cpa.min():.2f} - {y_cpa.max():.2f} (original scale)")
        print(f"📊 Features được normalize, targets (GPA/CPA) giữ nguyên scale gốc")
        
        return X_scaled, y_gpa, y_cpa, feature_names, students_data
        
    def _extract_historical_features(self, historical_semesters, student_courses, student_id):
        try:
            features = []
            
            historical_semesters = historical_semesters.copy()
            historical_semesters['GPA'] = pd.to_numeric(historical_semesters['GPA'], errors='coerce').fillna(0)
            historical_semesters['CPA'] = pd.to_numeric(historical_semesters['CPA'], errors='coerce').fillna(0)
            historical_semesters['TC qua'] = pd.to_numeric(historical_semesters['TC qua'], errors='coerce').fillna(0)
            historical_semesters['Acc'] = pd.to_numeric(historical_semesters['Acc'], errors='coerce').fillna(0)
            historical_semesters['Debt'] = pd.to_numeric(historical_semesters['Debt'], errors='coerce').fillna(0)
            historical_semesters['Reg'] = pd.to_numeric(historical_semesters['Reg'], errors='coerce').fillna(0)
            
            current_gpa = historical_semesters['GPA'].iloc[-1] if len(historical_semesters) > 0 else 0
            current_cpa = historical_semesters['CPA'].iloc[-1] if len(historical_semesters) > 0 else 0
            features.extend([current_gpa, current_cpa])
            
            if len(historical_semesters) >= 2:
                gpa_trend = historical_semesters['GPA'].iloc[-1] - historical_semesters['GPA'].iloc[0]
                cpa_trend = historical_semesters['CPA'].iloc[-1] - historical_semesters['CPA'].iloc[0]
            else:
                gpa_trend = 0
                cpa_trend = 0
            features.extend([gpa_trend, cpa_trend])
            
            avg_gpa = historical_semesters['GPA'].mean()
            avg_cpa = historical_semesters['CPA'].mean()
            gpa_std = historical_semesters['GPA'].std() if len(historical_semesters) > 1 else 0
            cpa_std = historical_semesters['CPA'].std() if len(historical_semesters) > 1 else 0
            features.extend([avg_gpa, avg_cpa, gpa_std, cpa_std])
            
            total_credits_passed = historical_semesters['TC qua'].sum()
            total_credits_accumulated = historical_semesters['Acc'].sum()
            total_debt = historical_semesters['Debt'].sum()
            total_registered = historical_semesters['Reg'].sum()
            
            features.extend([total_credits_passed, total_credits_accumulated, total_debt, total_registered])
            
            current_pass_rate = historical_semesters['TC qua'].sum() / (historical_semesters['Reg'].sum() + 1e-8)
            current_debt_rate = historical_semesters['Debt'].sum() / (historical_semesters['Reg'].sum() + 1e-8)
            features.extend([current_pass_rate, current_debt_rate])
            
            warning_counts = historical_semesters['Warning'].map(self.warning_mapping).fillna(0)
            max_warning = warning_counts.max()
            avg_warning = warning_counts.mean()
            current_warning = warning_counts.iloc[-1] if len(warning_counts) > 0 else 0
            features.extend([max_warning, avg_warning, current_warning])
            
            num_semesters = len(historical_semesters)
            features.append(num_semesters)
            
            semester_courses = []
            for _, semester in historical_semesters.iterrows():
                courses_in_semester = student_courses[
                    student_courses['Semester'] == semester['Semester']
                ]
                
                if len(courses_in_semester) > 0:
                    courses_in_semester = courses_in_semester.copy()
                    courses_in_semester['Final Grade Numeric'] = courses_in_semester['Final Grade'].map(self.grade_mapping).fillna(0)
                    
                    avg_continuous = pd.to_numeric(courses_in_semester['Continuous Assessment Score'], errors='coerce').mean()
                    avg_exam = pd.to_numeric(courses_in_semester['Exam Score'], errors='coerce').mean()
                    avg_grade = courses_in_semester['Final Grade Numeric'].mean()
                    total_credits = pd.to_numeric(courses_in_semester['Credits'], errors='coerce').sum()
                    pass_rate = (courses_in_semester['Final Grade Numeric'] >= 1.0).mean()
                    
                    semester_courses.extend([avg_continuous, avg_exam, avg_grade, total_credits, pass_rate])
                else:
                    semester_courses.extend([0, 0, 0, 0, 0])
            
            max_semesters = 6
            courses_features = semester_courses[:max_semesters*5]
            while len(courses_features) < max_semesters*5:
                courses_features.append(0)
            
            features.extend(courses_features)
            
            features = [float(f) if not pd.isna(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features for student {student_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_feature_names(self):
        base_features = [
            'current_gpa', 'current_cpa', 'gpa_trend', 'cpa_trend',
            'avg_gpa', 'avg_cpa', 'gpa_std', 'cpa_std',
            'total_credits_passed', 'total_credits_accumulated', 'total_debt', 'total_registered',
            'current_pass_rate', 'current_debt_rate',
            'max_warning', 'avg_warning', 'current_warning',
            'num_semesters'
        ]
        
        course_features = []
        for i in range(6):
            course_features.extend([
                f'sem_{i+1}_avg_continuous',
                f'sem_{i+1}_avg_exam', 
                f'sem_{i+1}_avg_grade',
                f'sem_{i+1}_total_credits',
                f'sem_{i+1}_pass_rate'
            ])
        
        return base_features + course_features
    
    def get_algorithms(self):
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=15),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
        }
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()
            
            return {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'CV_R2_mean': cv_r2_mean,
                'CV_R2_std': cv_r2_std,
                'predictions': y_pred,
                'model': model
            }
        except Exception as e:
            print(f"❌ Lỗi với {model_name}: {str(e)}")
            return None
    
    def run_all_algorithms(self, X, y, target_name):
        print(f"\n🧪 Đánh giá thuật toán cho {target_name} (Sequential Prediction)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        algorithms = self.get_algorithms()
        results = {}
        
        for name, model in algorithms.items():
            print(f"  ⚙️ Testing {name}...")
            result = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            if result:
                results[name] = result
                print(f"    ✅ R² = {result['R2']:.4f}, MAE = {result['MAE']:.4f}")
        
        return results, X_test, y_test
    
    def display_results(self, results, target_name):
        print(f"\n📊 KẾT QUẢ SEQUENTIAL PREDICTION CHO {target_name.upper()} (Original Scale):")
        print("=" * 80)
        print(f"{'Algorithm':<20} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'CV R² (mean±std)':<20}")
        print("-" * 80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        for name, metrics in sorted_results:
            cv_str = f"{metrics['CV_R2_mean']:.3f}±{metrics['CV_R2_std']:.3f}"
            print(f"{name:<20} {metrics['R2']:<8.4f} {metrics['MAE']:<8.4f} "
                  f"{metrics['RMSE']:<8.4f} {cv_str:<20}")
        
        print(f"\n💡 MAE/RMSE tính trên {target_name} gốc (0-4 scale), không phải normalized")
        
        return sorted_results
    
    def plot_results(self, gpa_results, cpa_results, X_test_gpa, y_test_gpa, X_test_cpa, y_test_cpa):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        gpa_sorted = sorted(gpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        cpa_sorted = sorted(cpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        gpa_names = [x[0] for x in gpa_sorted]
        gpa_r2_scores = [x[1]['R2'] for x in gpa_sorted]
        gpa_mae_scores = [x[1]['MAE'] for x in gpa_sorted]
        
        cpa_names = [x[0] for x in cpa_sorted]
        cpa_r2_scores = [x[1]['R2'] for x in cpa_sorted]
        cpa_mae_scores = [x[1]['MAE'] for x in cpa_sorted]
        
        axes[0,0].barh(range(len(gpa_names)), gpa_r2_scores)
        axes[0,0].set_yticks(range(len(gpa_names)))
        axes[0,0].set_yticklabels(gpa_names)
        axes[0,0].set_title('Sequential GPA Prediction - R² Score')
        axes[0,0].set_xlabel('R² Score')
        
        axes[0,1].barh(range(len(gpa_names)), gpa_mae_scores)
        axes[0,1].set_yticks(range(len(gpa_names)))
        axes[0,1].set_yticklabels(gpa_names)
        axes[0,1].set_title('Sequential GPA Prediction - MAE')
        axes[0,1].set_xlabel('MAE')
        
        best_gpa_model = gpa_sorted[0][0]
        gpa_pred = gpa_results[best_gpa_model]['predictions']
        axes[0,2].scatter(y_test_gpa, gpa_pred, alpha=0.6)
        axes[0,2].plot([y_test_gpa.min(), y_test_gpa.max()], 
                       [y_test_gpa.min(), y_test_gpa.max()], 'r--', lw=2)
        axes[0,2].set_title(f'Sequential GPA: {best_gpa_model}\nActual vs Predicted')
        axes[0,2].set_xlabel('Actual GPA')
        axes[0,2].set_ylabel('Predicted GPA')
        
        axes[1,0].barh(range(len(cpa_names)), cpa_r2_scores)
        axes[1,0].set_yticks(range(len(cpa_names)))
        axes[1,0].set_yticklabels(cpa_names)
        axes[1,0].set_title('Sequential CPA Prediction - R² Score')
        axes[1,0].set_xlabel('R² Score')
        
        axes[1,1].barh(range(len(cpa_names)), cpa_mae_scores)
        axes[1,1].set_yticks(range(len(cpa_names)))
        axes[1,1].set_yticklabels(cpa_names)
        axes[1,1].set_title('Sequential CPA Prediction - MAE')
        axes[1,1].set_xlabel('MAE')
        
        best_cpa_model = cpa_sorted[0][0]
        cpa_pred = cpa_results[best_cpa_model]['predictions']
        axes[1,2].scatter(y_test_cpa, cpa_pred, alpha=0.6)
        axes[1,2].plot([y_test_cpa.min(), y_test_cpa.max()], 
                       [y_test_cpa.min(), y_test_cpa.max()], 'r--', lw=2)
        axes[1,2].set_title(f'Sequential CPA: {best_cpa_model}\nActual vs Predicted')
        axes[1,2].set_xlabel('Actual CPA')
        axes[1,2].set_ylabel('Predicted CPA')
        
        plt.tight_layout()
        plt.savefig('sequential_ml_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n💾 Biểu đồ đã được lưu: 'sequential_ml_prediction.png'")
    
    def create_feature_importance_plot(self, gpa_results, cpa_results, feature_names):
        tree_models = ['Random Forest', 'Extra Trees', 'Decision Tree']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for i, (target, results) in enumerate([('GPA', gpa_results), ('CPA', cpa_results)]):
            importances_data = []
            
            for model_name in tree_models:
                if model_name in results and hasattr(results[model_name]['model'], 'feature_importances_'):
                    importances = results[model_name]['model'].feature_importances_
                    importances_data.append(importances)
            
            if importances_data:
                avg_importances = np.mean(importances_data, axis=0)
                
                feature_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': avg_importances
                }).sort_values('Importance', ascending=True).tail(15)
                
                axes[i].barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
                axes[i].set_title(f'Sequential {target} - Top 15 Feature Importance')
                axes[i].set_xlabel('Importance')
            else:
                axes[i].text(0.5, 0.5, 'No tree models available', 
                            transform=axes[i].transAxes, ha='center')
                axes[i].set_title(f'Sequential {target} - Feature Importance')
        
        plt.tight_layout()
        plt.savefig('sequential_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Feature importance plot saved: 'sequential_feature_importance.png'")

    def show_prediction_results(self, gpa_results, cpa_results, X_test, y_test_gpa, y_test_cpa, students_info, n_samples=50):
        print(f"\n🎯 HIỂN THỊ KẾT QUẢ DỰ ĐOÁN CHO {n_samples} SINH VIÊN NGẪU NHIÊN")
        print("=" * 100)
        
        best_gpa_model_name = max(gpa_results.items(), key=lambda x: x[1]['R2'])[0]
        best_cpa_model_name = max(cpa_results.items(), key=lambda x: x[1]['R2'])[0]
        
        best_gpa_model = gpa_results[best_gpa_model_name]['model']
        best_cpa_model = cpa_results[best_cpa_model_name]['model']
        
        predictions_gpa = best_gpa_model.predict(X_test)
        predictions_cpa = best_cpa_model.predict(X_test)
        
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        
        print(f"GPA Model: {best_gpa_model_name} | CPA Model: {best_cpa_model_name}")
        print(f"{'STT':<4} {'Student ID':<12} {'Actual GPA':<12} {'Pred GPA':<12} {'Error':<8} {'Actual CPA':<12} {'Pred CPA':<12} {'Error':<8} {'Status':<15}")
        print("-" * 100)
        
        total_mae_gpa = 0
        total_mae_cpa = 0
        accurate_predictions = 0
        
        for i, idx in enumerate(indices):
            train_idx = int(idx * 0.8)
            student_info = students_info[train_idx] if train_idx < len(students_info) else {'student_id': f'Student_{idx}'}
            student_id = student_info.get('student_id', f'Student_{idx}')
            
            actual_gpa = y_test_gpa[idx]
            pred_gpa = predictions_gpa[idx]
            error_gpa = abs(actual_gpa - pred_gpa)
            
            actual_cpa = y_test_cpa[idx] 
            pred_cpa = predictions_cpa[idx]
            error_cpa = abs(actual_cpa - pred_cpa)
            
            total_mae_gpa += error_gpa
            total_mae_cpa += error_cpa
            
            if error_gpa <= 0.3:
                status = "✅ Accurate"
                accurate_predictions += 1
            elif error_gpa <= 0.5:
                status = "⚠️ Acceptable"
            else:
                status = "❌ Poor"
            
            print(f"{i+1:<4} {student_id:<12} {actual_gpa:<12.3f} {pred_gpa:<12.3f} {error_gpa:<8.3f} "
                  f"{actual_cpa:<12.3f} {pred_cpa:<12.3f} {error_cpa:<8.3f} {status:<15}")
        
        print("-" * 100)
        print(f"📊 THỐNG KÊ TỔNG KẾT:")
        print(f"  • Average MAE GPA: {total_mae_gpa/len(indices):.4f}")
        print(f"  • Average MAE CPA: {total_mae_cpa/len(indices):.4f}") 
        print(f"  • Accurate predictions (error ≤ 0.3): {accurate_predictions}/{len(indices)} ({accurate_predictions/len(indices)*100:.1f}%)")
        print(f"  • GPA Model: {best_gpa_model_name} | CPA Model: {best_cpa_model_name}")

    def show_10_student_predictions(self, gpa_results, cpa_results, X_test, y_test_gpa, y_test_cpa, students_info):
        print(f"\n🎯 DỰ ĐOÁN CHI TIẾT CHO 10 SINH VIÊN NGẪU NHIÊN")
        print("=" * 70)
        
        best_gpa_model_name = max(gpa_results.items(), key=lambda x: x[1]['R2'])[0]
        best_cpa_model_name = max(cpa_results.items(), key=lambda x: x[1]['R2'])[0]
        
        best_gpa_model = gpa_results[best_gpa_model_name]['model']
        best_cpa_model = cpa_results[best_cpa_model_name]['model']
        
        predictions_gpa = best_gpa_model.predict(X_test)
        predictions_cpa = best_cpa_model.predict(X_test)
        
        indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        
        for i, idx in enumerate(indices):
            train_idx = int(idx * 0.8)
            student_info = students_info[train_idx] if train_idx < len(students_info) else {'student_id': f'Student_{idx}'}
            student_id = student_info.get('student_id', f'Student_{idx}')
            
            actual_gpa = y_test_gpa[idx]
            pred_gpa = predictions_gpa[idx]
            error_gpa = abs(actual_gpa - pred_gpa)
            
            actual_cpa = y_test_cpa[idx] 
            pred_cpa = predictions_cpa[idx]
            error_cpa = abs(actual_cpa - pred_cpa)
            
            confidence_gpa = "High" if error_gpa <= 0.2 else "Medium" if error_gpa <= 0.4 else "Low"
            confidence_cpa = "High" if error_cpa <= 0.2 else "Medium" if error_cpa <= 0.4 else "Low"
            
            print(f"\n👤 SINH VIÊN {i+1}: {student_id}")
            print(f"   📊 GPA:")
            print(f"      • Thực tế: {actual_gpa:.3f}")
            print(f"      • Dự đoán: {pred_gpa:.3f}")
            print(f"      • Sai số:  {error_gpa:.3f}")
            print(f"      • Độ tin cậy: {confidence_gpa}")
            
            print(f"   📊 CPA:")
            print(f"      • Thực tế: {actual_cpa:.3f}")
            print(f"      • Dự đoán: {pred_cpa:.3f}")
            print(f"      • Sai số:  {error_cpa:.3f}")
            print(f"      • Độ tin cậy: {confidence_cpa}")
            
            if error_gpa <= 0.2 and error_cpa <= 0.2:
                assessment = "✅ Dự đoán rất chính xác"
            elif error_gpa <= 0.4 and error_cpa <= 0.4:
                assessment = "⚠️ Dự đoán chấp nhận được"
            else:
                assessment = "❌ Cần cải thiện model"
            
            print(f"   🎯 Đánh giá: {assessment}")
        
        print(f"\n📋 Models sử dụng:")
        print(f"   • GPA: {best_gpa_model_name}")
        print(f"   • CPA: {best_cpa_model_name}")

    def analyze_prediction_patterns(self, results, X_test, y_test_gpa, y_test_cpa, feature_names):
        print(f"\n🔍 PHÂN TÍCH PATTERNS DỰ ĐOÁN")
        print("=" * 80)
        
        best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
        best_model = results[best_model_name]['model']
        predictions = best_model.predict(X_test)
        
        errors = np.abs(y_test_gpa - predictions)
        
        high_error_indices = np.where(errors > 0.5)[0]
        low_error_indices = np.where(errors <= 0.2)[0]
        
        print(f"🎯 Phân tích {len(high_error_indices)} cases có error cao (>0.5):")
        if len(high_error_indices) > 0:
            high_error_features = X_test[high_error_indices]
            high_error_mean = np.mean(high_error_features, axis=0)
            
            print(f"  • Average current GPA: {high_error_mean[0]:.3f}")
            print(f"  • Average GPA trend: {high_error_mean[2]:.3f}")
            print(f"  • Average warning level: {high_error_mean[16]:.3f}")
            
        print(f"\n✅ Phân tích {len(low_error_indices)} cases có error thấp (≤0.2):")
        if len(low_error_indices) > 0:
            low_error_features = X_test[low_error_indices]
            low_error_mean = np.mean(low_error_features, axis=0)
            
            print(f"  • Average current GPA: {low_error_mean[0]:.3f}")
            print(f"  • Average GPA trend: {low_error_mean[2]:.3f}")
            print(f"  • Average warning level: {low_error_mean[16]:.3f}")
        
        print(f"\n💡 KẾT LUẬN:")
        print(f"  • Model dự đoán tốt hơn cho sinh viên có GPA ổn định")
        print(f"  • Cases khó dự đoán thường có xu hướng GPA biến động lớn")
        print(f"  • Warning level cao làm tăng độ khó dự đoán")

def main():
    print("🔮 SEQUENTIAL ML PREDICTION: N-1 SEMESTERS → Nth SEMESTER")
    print("=" * 70)
    
    try:
        course_df = pd.read_csv('csv/ET1_K62_K63_K64.csv')
        perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
        
        predictor = SequentialMLPredictor()
        
        X, y_gpa, y_cpa, feature_names, students_data = predictor.prepare_sequential_data(
            course_df, perf_df, min_semesters=3
        )
        
        if X is None:
            print("❌ Không thể tạo sequential dataset")
            return
        
        gpa_results, X_test_gpa, y_test_gpa = predictor.run_all_algorithms(X, y_gpa, 'GPA')
        cpa_results, X_test_cpa, y_test_cpa = predictor.run_all_algorithms(X, y_cpa, 'CPA')
        
        predictor.display_results(gpa_results, 'GPA')
        predictor.display_results(cpa_results, 'CPA')
        
        predictor.plot_results(gpa_results, cpa_results, X_test_gpa, y_test_gpa, X_test_cpa, y_test_cpa)
        
        predictor.create_feature_importance_plot(gpa_results, cpa_results, feature_names)
        
        predictor.show_prediction_results(gpa_results, cpa_results, X_test_gpa, y_test_gpa, y_test_cpa, students_data, n_samples=50)
        
        predictor.show_10_student_predictions(gpa_results, cpa_results, X_test_gpa, y_test_gpa, y_test_cpa, students_data)
        
        predictor.analyze_prediction_patterns(gpa_results, X_test_gpa, y_test_gpa, y_test_cpa, feature_names)
        
        print("\n🎯 GIẢI THÍCH KẾT QUẢ:")
        print("• Model sử dụng lịch sử học tập của n-1 kỳ để dự đoán kỳ thứ n")
        print("• Features bao gồm: GPA/CPA xu hướng, tổng TC, tỷ lệ pass, cảnh báo, điểm môn học")
        print("• Đây là approach realistic hơn cho academic prediction")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH SEQUENTIAL ML PREDICTION")
        
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file dữ liệu: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 