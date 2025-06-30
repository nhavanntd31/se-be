#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_processor import (
    StudentPerformanceDataProcessor,
    load_and_prepare_data,
    load_and_prepare_temporal_data,
    compute_denormalized_metrics
)

class TraditionalMLTrainer:
    def __init__(self, processor: StudentPerformanceDataProcessor):
        self.processor = processor
        self.models = {}
        self.trained_models = {}
        
    def prepare_ml_data(self, sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        targets = []
        
        for seq in sequences:
            student_features = []
            
            for sem_data in seq['semesters']:
                course_features = np.array(sem_data['courses'])
                if len(course_features) > 0:
                    course_stats = [
                        np.mean(course_features, axis=0),
                        np.std(course_features, axis=0),
                        np.min(course_features, axis=0),
                        np.max(course_features, axis=0),
                        np.array([len(course_features)])
                    ]
                    course_aggregated = np.concatenate([np.array(stat).flatten() for stat in course_stats])
                else:
                    course_aggregated = np.zeros(29)
                
                performance_features = np.array(sem_data['performance'])
                combined_features = np.concatenate([course_aggregated, performance_features])
                student_features.extend(combined_features)
            
            if len(student_features) > 0:
                max_length = 400
                if len(student_features) > max_length:
                    student_features = student_features[:max_length]
                else:
                    student_features.extend([0] * (max_length - len(student_features)))
                
                features.append(student_features)
                targets.append([seq['target_gpa'], seq['target_cpa']])
        
        return np.array(features), np.array(targets)
    
    def initialize_models(self):
        self.models = {
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )),
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.1, random_state=42)),
            'SVM': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale'))
        }
    
    def evaluate_model_with_denormalization(self, model, X_test, y_test_normalized):
        y_pred_normalized = model.predict(X_test)
        
        normalized_metrics = {
            'gpa_mse': mean_squared_error(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_mse': mean_squared_error(y_test_normalized[:, 1], y_pred_normalized[:, 1]),
            'gpa_mae': mean_absolute_error(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_mae': mean_absolute_error(y_test_normalized[:, 1], y_pred_normalized[:, 1]),
            'gpa_r2': r2_score(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_r2': r2_score(y_test_normalized[:, 1], y_pred_normalized[:, 1])
        }
        
        denormalized_metrics = compute_denormalized_metrics(
            y_pred_normalized, y_test_normalized, self.processor
        )
        
        return {**normalized_metrics, **denormalized_metrics}
    
    def train_and_evaluate_all(self, train_sequences, test_sequences):
        print("\n" + "="*60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        X_train, y_train = self.prepare_ml_data(train_sequences)
        X_test, y_test = self.prepare_ml_data(test_sequences)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        self.initialize_models()
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                metrics = self.evaluate_model_with_denormalization(model, X_test, y_test)
                metrics['training_time'] = training_time
                
                results[model_name] = metrics
                self.trained_models[model_name] = model
                
                print(f"  Training time: {training_time:.2f}s")
                print(f"  GPA (Normalized) - MSE: {metrics['gpa_mse']:.4f}, MAE: {metrics['gpa_mae']:.4f}, R²: {metrics['gpa_r2']:.4f}")
                print(f"  CPA (Normalized) - MSE: {metrics['cpa_mse']:.4f}, MAE: {metrics['cpa_mae']:.4f}, R²: {metrics['cpa_r2']:.4f}")
                
                if 'gpa_mse_denorm' in metrics and metrics['gpa_mse_denorm'] != float('inf'):
                    print(f"  GPA (Denormalized) - MSE: {metrics['gpa_mse_denorm']:.4f}, MAE: {metrics['gpa_mae_denorm']:.4f}, R²: {metrics['gpa_r2_denorm']:.4f}")
                    print(f"  CPA (Denormalized) - MSE: {metrics['cpa_mse_denorm']:.4f}, MAE: {metrics['cpa_mae_denorm']:.4f}, R²: {metrics['cpa_r2_denorm']:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def save_best_models(self, results):
        best_gpa_model = None
        best_cpa_model = None
        best_gpa_mae = float('inf')
        best_cpa_mae = float('inf')
        
        for model_name, metrics in results.items():
            if metrics is None:
                continue
                
            if 'gpa_mae_denorm' in metrics and metrics['gpa_mae_denorm'] < best_gpa_mae:
                best_gpa_mae = metrics['gpa_mae_denorm']
                best_gpa_model = (model_name, self.trained_models[model_name])
            
            if 'cpa_mae_denorm' in metrics and metrics['cpa_mae_denorm'] < best_cpa_mae:
                best_cpa_mae = metrics['cpa_mae_denorm']
                best_cpa_model = (model_name, self.trained_models[model_name])
        
        if best_gpa_model:
            with open('best_gpa_model.pkl', 'wb') as f:
                pickle.dump(best_gpa_model[1], f)
            print(f"\nBest GPA model ({best_gpa_model[0]}) saved to 'best_gpa_model.pkl'")
            print(f"GPA MAE (denormalized): {best_gpa_mae:.4f}")
        
        if best_cpa_model:
            with open('best_cpa_model.pkl', 'wb') as f:
                pickle.dump(best_cpa_model[1], f)
            print(f"Best CPA model ({best_cpa_model[0]}) saved to 'best_cpa_model.pkl'")
            print(f"CPA MAE (denormalized): {best_cpa_mae:.4f}")
        
        return best_gpa_model, best_cpa_model

    def show_sample_predictions(self, test_sequences, num_samples=15):
        if not self.trained_models:
            print("No trained models available for predictions.")
            return
            
        X_test, y_test = self.prepare_ml_data(test_sequences)
        
        best_model_name = None
        best_model = None
        best_mae = float('inf')
        
        for model_name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            if mae < best_mae:
                best_mae = mae
                best_model_name = model_name
                best_model = model
        
        if best_model is None:
            print("No valid model found for predictions.")
            return
            
        y_pred_normalized = best_model.predict(X_test)
        
        y_test_denorm = self.processor.denormalize_performance(y_test)
        y_pred_denorm = self.processor.denormalize_performance(y_pred_normalized)
        
        print(f"\n" + "="*100)
        print(f"SAMPLE PREDICTIONS - BEST MODEL: {best_model_name}")
        print("="*100)
        print(f"{'Student':<15} {'Sequence Type':<25} {'Actual GPA':<12} {'Pred GPA':<12} {'GPA Error':<12} {'Actual CPA':<12} {'Pred CPA':<12} {'CPA Error':<12}")
        print("-" * 100)
        
        num_samples = min(num_samples, len(y_test_denorm))
        indices = np.random.choice(len(y_test_denorm), num_samples, replace=False)
        
        total_gpa_error = 0
        total_cpa_error = 0
        
        for i, idx in enumerate(indices):
            actual_gpa = y_test_denorm[idx, 0]
            pred_gpa = y_pred_denorm[idx, 0]
            gpa_error = abs(actual_gpa - pred_gpa)
            
            actual_cpa = y_test_denorm[idx, 1]
            pred_cpa = y_pred_denorm[idx, 1]
            cpa_error = abs(actual_cpa - pred_cpa)
            
            total_gpa_error += gpa_error
            total_cpa_error += cpa_error
            
            sequence_type = test_sequences[idx].get('sequence_type', 'N/A')[:24]
            student_id = str(test_sequences[idx].get('student_id', f'SV_{i+1}'))[:14]
            
            print(f"{student_id:<15} {sequence_type:<25} {actual_gpa:<12.3f} {pred_gpa:<12.3f} {gpa_error:<12.3f} "
                  f"{actual_cpa:<12.3f} {pred_cpa:<12.3f} {cpa_error:<12.3f}")
        
        avg_gpa_error = total_gpa_error / num_samples
        avg_cpa_error = total_cpa_error / num_samples
        
        print("-" * 100)
        print(f"{'Average':<15} {'':<25} {'':<12} {'':<12} {avg_gpa_error:<12.3f} {'':<12} {'':<12} {avg_cpa_error:<12.3f}")
        print(f"\nAverage absolute error on sample:")
        print(f"  GPA: {avg_gpa_error:.3f}")
        print(f"  CPA: {avg_cpa_error:.3f}")
        
        sequence_error_stats = {}
        for idx in indices:
            seq_type = test_sequences[idx].get('sequence_type', 'N/A')
            actual_gpa = y_test_denorm[idx, 0]
            pred_gpa = y_pred_denorm[idx, 0]
            actual_cpa = y_test_denorm[idx, 1]
            pred_cpa = y_pred_denorm[idx, 1]
            
            gpa_error = abs(actual_gpa - pred_gpa)
            cpa_error = abs(actual_cpa - pred_cpa)
            
            if seq_type not in sequence_error_stats:
                sequence_error_stats[seq_type] = {'gpa_errors': [], 'cpa_errors': []}
            sequence_error_stats[seq_type]['gpa_errors'].append(gpa_error)
            sequence_error_stats[seq_type]['cpa_errors'].append(cpa_error)
        
        print(f"\nError by sequence type:")
        for seq_type, stats in sequence_error_stats.items():
            avg_gpa_err = np.mean(stats['gpa_errors'])
            avg_cpa_err = np.mean(stats['cpa_errors'])
            count = len(stats['gpa_errors'])
            print(f"  {seq_type}: GPA {avg_gpa_err:.3f}, CPA {avg_cpa_err:.3f} (n={count})")

def plot_ml_comparison(ml_results):
    models = list(ml_results.keys())
    models = [m for m in models if ml_results[m] is not None]
    
    gpa_mae_denorm = [ml_results[m]['gpa_mae_denorm'] for m in models if 'gpa_mae_denorm' in ml_results[m]]
    cpa_mae_denorm = [ml_results[m]['cpa_mae_denorm'] for m in models if 'cpa_mae_denorm' in ml_results[m]]
    gpa_mse_denorm = [ml_results[m]['gpa_mse_denorm'] for m in models if 'gpa_mse_denorm' in ml_results[m]]
    cpa_mse_denorm = [ml_results[m]['cpa_mse_denorm'] for m in models if 'cpa_mse_denorm' in ml_results[m]]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars1 = ax1.bar(models, gpa_mae_denorm, color=colors, alpha=0.7)
    ax1.set_title('GPA Mean Absolute Error (Denormalized)')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, gpa_mae_denorm):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gpa_mae_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars2 = ax2.bar(models, cpa_mae_denorm, color=colors, alpha=0.7)
    ax2.set_title('CPA Mean Absolute Error (Denormalized)')
    ax2.set_ylabel('MAE')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, cpa_mae_denorm):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpa_mae_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars3 = ax3.bar(models, gpa_mse_denorm, color=colors, alpha=0.7)
    ax3.set_title('GPA Mean Squared Error (Denormalized)')
    ax3.set_ylabel('MSE')
    ax3.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars3, gpa_mse_denorm):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gpa_mse_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars4 = ax4.bar(models, cpa_mse_denorm, color=colors, alpha=0.7)
    ax4.set_title('CPA Mean Squared Error (Denormalized)')
    ax4.set_ylabel('MSE')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, cpa_mse_denorm):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpa_mse_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ml_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSo sánh các mô hình đã được lưu vào 'ml_algorithms_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description='Traditional ML Models for Student Performance Prediction')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV file')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of students for training (default: 0.6)')
    parser.add_argument('--use_temporal_split', action='store_true', help='Use temporal split to avoid data leak')
    
    args = parser.parse_args()
    
    print("=== TRADITIONAL ML MODELS FOR STUDENT PERFORMANCE PREDICTION ===")
    print("Models: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVM")
    print("All metrics computed on original scale (denormalized)")
    
    if args.use_temporal_split:
        print("Using TEMPORAL SPLIT (no data leak)")
        print("Train: Kỳ 1-4 dự đoán kỳ 5")
        print("Test: Kỳ 1-4→5, 2-5→6, 3-6→7, ... từ sinh viên khác")
    else:
        print("Using RANDOM SPLIT (may have data leak)")
    
    try:
        if args.use_temporal_split:
            train_sequences, test_sequences, processor = load_and_prepare_temporal_data(
                args.course_csv, args.performance_csv, args.train_ratio
            )
        else:
            sequences, processor = load_and_prepare_data(args.course_csv, args.performance_csv)
            train_sequences, test_sequences = train_test_split(sequences, test_size=0.4, random_state=42)
            print(f"Train: {len(train_sequences)}, Test: {len(test_sequences)}")
        
        ml_trainer = TraditionalMLTrainer(processor)
        ml_results = ml_trainer.train_and_evaluate_all(train_sequences, test_sequences)
        
        best_gpa_model, best_cpa_model = ml_trainer.save_best_models(ml_results)
        
        if args.use_temporal_split:
            ml_trainer.show_sample_predictions(test_sequences, num_samples=20)
        
        processor.save_scalers()
        
        print(f"\n" + "="*80)
        print("FINAL RESULTS (DENORMALIZED METRICS)")
        print("="*80)
        
        print(f"{'Model':<20} {'GPA MAE':<10} {'GPA MSE':<10} {'GPA R²':<10} {'CPA MAE':<10} {'CPA MSE':<10} {'CPA R²':<10} {'Time(s)':<10}")
        print("-" * 90)
        
        for model_name, metrics in ml_results.items():
            if metrics is not None and 'gpa_mae_denorm' in metrics:
                print(f"{model_name:<20} {metrics['gpa_mae_denorm']:<10.4f} {metrics['gpa_mse_denorm']:<10.4f} "
                      f"{metrics['gpa_r2_denorm']:<10.4f} {metrics['cpa_mae_denorm']:<10.4f} "
                      f"{metrics['cpa_mse_denorm']:<10.4f} {metrics['cpa_r2_denorm']:<10.4f} {metrics['training_time']:<10.2f}")
        
        plot_ml_comparison(ml_results)
        
        print(f"\nFiles saved:")
        print(f"  - Best GPA model: 'best_gpa_model.pkl' ({best_gpa_model[0] if best_gpa_model else 'None'})")
        print(f"  - Best CPA model: 'best_cpa_model.pkl' ({best_cpa_model[0] if best_cpa_model else 'None'})")
        print(f"  - Performance scaler: 'feature_scaler.pkl'")
        print(f"  - Course scaler: 'course_scaler.pkl'")
        print(f"  - Comparison plot: 'ml_algorithms_comparison.png'")
        
        if best_gpa_model:
            print(f"\nBest GPA model: {best_gpa_model[0]}")
        if best_cpa_model:
            print(f"Best CPA model: {best_cpa_model[0]}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 