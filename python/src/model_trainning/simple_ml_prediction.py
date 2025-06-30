import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleMLPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_and_prepare_data(self):
        print("🔄 Đang tải và chuẩn bị dữ liệu...")
        
        try:
            perf_df = pd.read_csv('csv/ET1_K62_K63_K64_performance.csv')
            
            print(f"📊 Dữ liệu gốc: {len(perf_df):,} records")
            print(f"📊 Columns: {list(perf_df.columns)}")
            
            numeric_features = []
            for col in perf_df.columns:
                if col not in ['student_id', 'Semester', 'Level', 'Warning']:
                    perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')
                    if col not in ['GPA', 'CPA']:
                        numeric_features.append(col)
            
            feature_cols = numeric_features
            print(f"📊 Features được sử dụng: {feature_cols}")
            
            df_clean = perf_df.dropna(subset=['GPA', 'CPA'])
            print(f"📊 Sau khi loại bỏ NaN GPA/CPA: {len(df_clean):,} records")
            
            X = df_clean[feature_cols]
            y_gpa = df_clean['GPA']
            y_cpa = df_clean['CPA']
            
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            print(f"📊 Final dataset: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")
            print(f"📊 GPA range: {y_gpa.min():.2f} - {y_gpa.max():.2f} (original scale)")
            print(f"📊 CPA range: {y_cpa.min():.2f} - {y_cpa.max():.2f} (original scale)")
            print(f"📊 Features được normalize, targets (GPA/CPA) giữ nguyên scale gốc")
            
            return X_scaled, y_gpa.values, y_cpa.values, feature_cols
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            return None, None, None, None
    
    def get_algorithms(self):
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
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
        print(f"\n🧪 Đánh giá thuật toán cho {target_name}...")
        
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
        print(f"\n📊 KẾT QUẢ CHO {target_name.upper()}:")
        print("=" * 80)
        print(f"{'Algorithm':<20} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'CV R² (mean±std)':<20}")
        print("-" * 80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        for name, metrics in sorted_results:
            cv_str = f"{metrics['CV_R2_mean']:.3f}±{metrics['CV_R2_std']:.3f}"
            print(f"{name:<20} {metrics['R2']:<8.4f} {metrics['MAE']:<8.4f} "
                  f"{metrics['RMSE']:<8.4f} {cv_str:<20}")
        
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
        axes[0,0].set_title('GPA - R² Score')
        axes[0,0].set_xlabel('R² Score')
        
        axes[0,1].barh(range(len(gpa_names)), gpa_mae_scores)
        axes[0,1].set_yticks(range(len(gpa_names)))
        axes[0,1].set_yticklabels(gpa_names)
        axes[0,1].set_title('GPA - MAE')
        axes[0,1].set_xlabel('MAE')
        
        best_gpa_model = gpa_sorted[0][0]
        gpa_pred = gpa_results[best_gpa_model]['predictions']
        axes[0,2].scatter(y_test_gpa, gpa_pred, alpha=0.6)
        axes[0,2].plot([y_test_gpa.min(), y_test_gpa.max()], 
                       [y_test_gpa.min(), y_test_gpa.max()], 'r--', lw=2)
        axes[0,2].set_title(f'GPA: {best_gpa_model}\nActual vs Predicted')
        axes[0,2].set_xlabel('Actual GPA')
        axes[0,2].set_ylabel('Predicted GPA')
        
        axes[1,0].barh(range(len(cpa_names)), cpa_r2_scores)
        axes[1,0].set_yticks(range(len(cpa_names)))
        axes[1,0].set_yticklabels(cpa_names)
        axes[1,0].set_title('CPA - R² Score')
        axes[1,0].set_xlabel('R² Score')
        
        axes[1,1].barh(range(len(cpa_names)), cpa_mae_scores)
        axes[1,1].set_yticks(range(len(cpa_names)))
        axes[1,1].set_yticklabels(cpa_names)
        axes[1,1].set_title('CPA - MAE')
        axes[1,1].set_xlabel('MAE')
        
        best_cpa_model = cpa_sorted[0][0]
        cpa_pred = cpa_results[best_cpa_model]['predictions']
        axes[1,2].scatter(y_test_cpa, cpa_pred, alpha=0.6)
        axes[1,2].plot([y_test_cpa.min(), y_test_cpa.max()], 
                       [y_test_cpa.min(), y_test_cpa.max()], 'r--', lw=2)
        axes[1,2].set_title(f'CPA: {best_cpa_model}\nActual vs Predicted')
        axes[1,2].set_xlabel('Actual CPA')
        axes[1,2].set_ylabel('Predicted CPA')
        
        plt.tight_layout()
        plt.savefig('simple_ml_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n💾 Biểu đồ đã được lưu: 'simple_ml_comparison.png'")
    
    def save_best_models(self, gpa_results, cpa_results):
        import joblib
        
        gpa_sorted = sorted(gpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        cpa_sorted = sorted(cpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        best_gpa_name, best_gpa_result = gpa_sorted[0]
        best_cpa_name, best_cpa_result = cpa_sorted[0]
        
        joblib.dump(best_gpa_result['model'], 'simple_best_gpa_model.pkl')
        joblib.dump(best_cpa_result['model'], 'simple_best_cpa_model.pkl')
        joblib.dump(self.scaler, 'simple_scaler.pkl')
        joblib.dump(self.imputer, 'simple_imputer.pkl')
        
        print(f"\n💾 Models đã được lưu:")
        print(f"  🎯 Best GPA model: {best_gpa_name} (R² = {best_gpa_result['R2']:.4f})")
        print(f"  🎯 Best CPA model: {best_cpa_name} (R² = {best_cpa_result['R2']:.4f})")

    def show_10_student_predictions(self, gpa_results, cpa_results, X_test, y_test_gpa, y_test_cpa):
        print(f"\n🎯 DỰ ĐOÁN CHI TIẾT CHO 10 SINH VIÊN NGẪU NHIÊN")
        print("=" * 70)
        
        gpa_sorted = sorted(gpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        cpa_sorted = sorted(cpa_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        best_gpa_name, best_gpa_result = gpa_sorted[0]
        best_cpa_name, best_cpa_result = cpa_sorted[0]
        
        best_gpa_model = best_gpa_result['model']
        best_cpa_model = best_cpa_result['model']
        
        predictions_gpa = best_gpa_model.predict(X_test)
        predictions_cpa = best_cpa_model.predict(X_test)
        
        indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        
        for i, idx in enumerate(indices):
            actual_gpa = y_test_gpa.iloc[idx] if hasattr(y_test_gpa, 'iloc') else y_test_gpa[idx]
            pred_gpa = predictions_gpa[idx]
            error_gpa = abs(actual_gpa - pred_gpa)
            
            actual_cpa = y_test_cpa.iloc[idx] if hasattr(y_test_cpa, 'iloc') else y_test_cpa[idx]
            pred_cpa = predictions_cpa[idx]
            error_cpa = abs(actual_cpa - pred_cpa)
            
            confidence_gpa = "High" if error_gpa <= 0.2 else "Medium" if error_gpa <= 0.4 else "Low"
            confidence_cpa = "High" if error_cpa <= 0.2 else "Medium" if error_cpa <= 0.4 else "Low"
            
            print(f"\n👤 SINH VIÊN {i+1}: Student_{idx}")
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
        print(f"   • GPA: {best_gpa_name}")
        print(f"   • CPA: {best_cpa_name}")

def create_feature_importance_plot(gpa_results, cpa_results, feature_names):
    tree_models = ['Random Forest', 'Extra Trees', 'Decision Tree']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (target, results) in enumerate([('GPA', gpa_results), ('CPA', cpa_results)]):
        importances_data = []
        
        for model_name in tree_models:
            if model_name in results and hasattr(results[model_name]['model'], 'feature_importances_'):
                importances = results[model_name]['model'].feature_importances_
                importances_data.append({
                    'Model': model_name,
                    'Importances': importances
                })
        
        if importances_data:
            avg_importances = np.mean([data['Importances'] for data in importances_data], axis=0)
            
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': avg_importances
            }).sort_values('Importance', ascending=True)
            
            axes[i].barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
            axes[i].set_title(f'{target} - Feature Importance (Average)')
            axes[i].set_xlabel('Importance')
        else:
            axes[i].text(0.5, 0.5, 'No tree models available', 
                        transform=axes[i].transAxes, ha='center')
            axes[i].set_title(f'{target} - Feature Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Feature importance plot saved: 'feature_importance.png'")

def main():
    print("🤖 SIMPLE MACHINE LEARNING FOR GPA/CPA PREDICTION")
    print("=" * 60)
    
    predictor = SimpleMLPredictor()
    
    X, y_gpa, y_cpa, feature_names = predictor.load_and_prepare_data()
    
    if X is None:
        print("❌ Không thể tải dữ liệu. Kết thúc chương trình.")
        return
    
    gpa_results, X_test_gpa, y_test_gpa = predictor.run_all_algorithms(X, y_gpa, 'GPA')
    cpa_results, X_test_cpa, y_test_cpa = predictor.run_all_algorithms(X, y_cpa, 'CPA')
    
    gpa_sorted = predictor.display_results(gpa_results, 'GPA')
    cpa_sorted = predictor.display_results(cpa_results, 'CPA')
    
    predictor.plot_results(gpa_results, cpa_results, X_test_gpa, y_test_gpa, X_test_cpa, y_test_cpa)
    
    create_feature_importance_plot(gpa_results, cpa_results, feature_names)
    
    predictor.save_best_models(gpa_results, cpa_results)
    
    predictor.show_10_student_predictions(gpa_results, cpa_results, X_test_gpa, y_test_gpa, y_test_cpa)
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH ĐÁNH GIÁ THUẬT TOÁN MACHINE LEARNING")

if __name__ == "__main__":
    main() 