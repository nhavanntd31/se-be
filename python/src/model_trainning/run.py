#!/usr/bin/env python3
"""
Script chạy nhanh cho Student Performance Prediction Model

Usage:
    python run.py --mode [install|analyze|train|demo]
"""

import sys
import argparse
import os

def install_dependencies():
    """Cài đặt dependencies"""
    print("🚀 Cài đặt dependencies...")
    os.system("python install_dependencies.py")

def analyze_data():
    """Phân tích dữ liệu"""
    print("📊 Phân tích dữ liệu...")
    os.system("python data_analysis.py")

def train_model():
    """Train mô hình"""
    print("🏋️ Bắt đầu training mô hình...")
    os.system("python student_performance_prediction.py")

def run_demo():
    """Chạy demo"""
    print("🎬 Chạy demo dự đoán...")
    os.system("python demo_prediction.py")

def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction Model Runner")
    parser.add_argument('--mode', choices=['install', 'analyze', 'train', 'demo', 'all'], 
                       required=True, help='Chọn mode để chạy')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE PREDICTION MODEL")
    print("=" * 60)
    
    if args.mode == 'install':
        install_dependencies()
    elif args.mode == 'analyze':
        analyze_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'all':
        print("🔄 Chạy toàn bộ pipeline...")
        install_dependencies()
        print("\n" + "="*40 + "\n")
        analyze_data()
        print("\n" + "="*40 + "\n")
        train_model()
        print("\n" + "="*40 + "\n")
        run_demo()
    
    print("\n✅ Hoàn thành!")

if __name__ == "__main__":
    main() 
