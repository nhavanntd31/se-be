#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def run_ml_models(course_csv: str, performance_csv: str) -> Dict:
    """Chạy traditional ML models và return results"""
    print("="*60)
    print("RUNNING TRADITIONAL ML MODELS")
    print("="*60)
    
    cmd = [
        sys.executable, 'predict_ml.py',
        '--course_csv', course_csv,
        '--performance_csv', performance_csv
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    ml_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running ML models: {result.stderr}")
        return None
    
    print(result.stdout)
    print(f"Total ML training time: {ml_time:.2f}s")
    
    return {'time': ml_time}

def run_transformer_model(course_csv: str, performance_csv: str, epochs: int = 50) -> Dict:
    """Chạy transformer model và return results"""
    print("\n" + "="*60)
    print("RUNNING TRANSFORMER MODEL")
    print("="*60)
    
    cmd = [
        sys.executable, 'train_transformer.py',
        '--course_csv', course_csv,
        '--performance_csv', performance_csv,
        '--epochs', str(epochs)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    transformer_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running Transformer model: {result.stderr}")
        return None
    
    print(result.stdout)
    print(f"Total Transformer training time: {transformer_time:.2f}s")
    
    return {'time': transformer_time}

def create_comparison_summary():
    """Tạo summary so sánh kết quả"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nFiles created:")
    print("├── Traditional ML:")
    print("│   ├── best_gpa_model.pkl")
    print("│   ├── best_cpa_model.pkl") 
    print("│   └── ml_algorithms_comparison.png")
    print("├── Transformer:")
    print("│   ├── best_model.pth")
    print("│   └── transformer_results.png")
    print("└── Shared:")
    print("    ├── feature_scaler.pkl")
    print("    └── course_scaler.pkl")
    
    print("\nNext steps:")
    print("1. Xem ml_algorithms_comparison.png để so sánh traditional ML models")
    print("2. Xem transformer_results.png để xem training progress của transformer")
    print("3. So sánh denormalized MAE/MSE values từ output")
    print("4. Chọn model tốt nhất dựa trên:")
    print("   - Performance metrics (MAE, MSE, R²)")
    print("   - Training time")
    print("   - Model complexity")
    print("   - Interpretability requirements")

def plot_time_comparison(ml_time: float, transformer_time: float):
    """Vẽ biểu đồ so sánh training time"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ['Traditional ML\n(6 models)', 'Transformer\n(Deep Learning)']
    times = [ml_time, transformer_time]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax.bar(models, times, color=colors, alpha=0.7)
    
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining time comparison saved to 'training_time_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description='Compare Traditional ML vs Transformer Models')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for transformer training')
    parser.add_argument('--skip_ml', action='store_true', help='Skip traditional ML training')
    parser.add_argument('--skip_transformer', action='store_true', help='Skip transformer training')
    
    args = parser.parse_args()
    
    print("=== MODEL COMPARISON SCRIPT ===")
    print("This script will run both traditional ML and transformer models")
    print("and provide a comprehensive comparison.")
    print(f"Course data: {args.course_csv}")
    print(f"Performance data: {args.performance_csv}")
    print(f"Transformer epochs: {args.epochs}")
    
    total_start_time = time.time()
    ml_result = None
    transformer_result = None
    
    try:
        # Run traditional ML models
        if not args.skip_ml:
            ml_result = run_ml_models(args.course_csv, args.performance_csv)
        else:
            print("\nSkipping traditional ML training...")
        
        # Run transformer model
        if not args.skip_transformer:
            transformer_result = run_transformer_model(args.course_csv, args.performance_csv, args.epochs)
        else:
            print("\nSkipping transformer training...")
        
        total_time = time.time() - total_start_time
        
        # Create comparison summary
        create_comparison_summary()
        
        # Plot time comparison if both models were run
        if ml_result and transformer_result:
            plot_time_comparison(ml_result['time'], transformer_result['time'])
            
            print(f"\n" + "="*50)
            print("TIMING SUMMARY")
            print("="*50)
            print(f"Traditional ML time: {ml_result['time']:.2f}s")
            print(f"Transformer time: {transformer_result['time']:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            
            if transformer_result['time'] > ml_result['time']:
                ratio = transformer_result['time'] / ml_result['time']
                print(f"Transformer took {ratio:.1f}x longer than traditional ML")
            else:
                ratio = ml_result['time'] / transformer_result['time']
                print(f"Traditional ML took {ratio:.1f}x longer than transformer")
        
        print(f"\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        print("1. If you need quick results: Use traditional ML (faster training)")
        print("2. If you need high accuracy: Compare denormalized MAE values")
        print("3. If you need interpretability: Traditional ML models are more interpretable")
        print("4. If you have lots of data: Transformer might perform better")
        print("5. For production: Consider training time vs accuracy trade-off")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 