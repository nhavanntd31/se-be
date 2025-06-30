"""
Student Performance Prediction Model Package

This package contains AI models for predicting student academic performance
using LSTM + Transformer architecture.

Main modules:
- student_performance_prediction: Main model and training pipeline
- demo_prediction: Demo script for testing predictions
- data_analysis: Exploratory data analysis tools
- install_dependencies: Dependencies installation script
"""

from .student_performance_prediction import (
    StudentPerformanceDataProcessor,
    StudentPerformancePredictor,
    ModelTrainer,
    create_student_sequences
)

from .demo_prediction import (
    predict_student_performance,
    analyze_student_trajectory
)

from .data_analysis import (
    analyze_course_data,
    analyze_performance_data,
    analyze_student_trajectories
)

__version__ = "1.0.0"
__author__ = "AI Team"
__description__ = "Student Performance Prediction using LSTM + Transformer" 