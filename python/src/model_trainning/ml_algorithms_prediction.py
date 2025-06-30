#!/usr/bin/env python3
import argparse
import sys
from student_performance_prediction import (
    StudentPerformanceDataProcessor, 
    create_student_sequences,
    TraditionalMLTrainer,
    plot_ml_comparison
)
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Traditional ML Models for Student Performance Prediction')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV file')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV file')
    
    args = parser.parse_args()
    
    print("=== TRADITIONAL ML MODELS FOR STUDENT PERFORMANCE PREDICTION ===")
    print("Models: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVM")
    print("All metrics computed on original scale (denormalized)")
    
    print(f"\nLoading data...")
    print(f"Course CSV: {args.course_csv}")
    print(f"Performance CSV: {args.performance_csv}")
    
    try:
        course_df = pd.read_csv(args.course_csv)
        perf_df = pd.read_csv(args.performance_csv)
        
        print(f"Course data: {len(course_df)} records")
        print(f"Performance data: {len(perf_df)} records")
        
        print(f"\nProcessing data...")
        processor = StudentPerformanceDataProcessor()
        sequences = create_student_sequences(course_df, perf_df, processor)
        
        print(f"Created {len(sequences)} training sequences")
        
        train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
        print(f"Train: {len(train_sequences)}, Test: {len(test_sequences)}")
        
        ml_trainer = TraditionalMLTrainer(processor)
        ml_results = ml_trainer.train_and_evaluate_all(train_sequences, test_sequences)
        
        best_gpa_model, best_cpa_model = ml_trainer.save_best_models(ml_results)
        
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
        
        plot_ml_comparison(ml_results, None)
        
        print(f"\nFiles saved:")
        print(f"  - Best GPA model: 'best_gpa_model.pkl' ({best_gpa_model[0] if best_gpa_model else 'None'})")
        print(f"  - Best CPA model: 'best_cpa_model.pkl' ({best_cpa_model[0] if best_cpa_model else 'None'})")
        print(f"  - Feature scaler: 'feature_scaler.pkl'")
        print(f"  - Comparison plot: 'ml_algorithms_comparison.png'")
        
        if best_gpa_model:
            print(f"\nBest GPA model: {best_gpa_model[0]}")
        if best_cpa_model:
            print(f"Best CPA model: {best_cpa_model[0]}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 