from datetime import datetime
import random
import uuid
from sqlalchemy.orm import Session
from models.models import Student, StudentProcess, StudentCourse, Department, Major, Class, Semester, StudentPrediction
from typing import List, Optional
from services.student_service import StudentService
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_trainning'))
from model_trainning.prediction_service import create_prediction_service

class StudentPredictService:
    def __init__(self, db: Session):
        logger.info("Initializing StudentPredictService")
        self.db = db
        self.student_service = StudentService(db)
        logger.info("Creating ML prediction service")
        self.prediction_service = create_prediction_service()
        logger.info("StudentPredictService initialization completed")

    def save_student_prediction(self, student_prediction: StudentPrediction):
        logger.info(f"Saving student prediction for student: {student_prediction.studentId}")
        self.db.add(student_prediction)
        self.db.commit()
        self.db.refresh(student_prediction)
        logger.info(f"Student prediction saved with ID: {student_prediction.id}")
        return student_prediction
    
    def save_batch_student_prediction(self, student_predictions: List[StudentPrediction]):
        logger.info(f"Saving batch of {len(student_predictions)} student predictions")
        print(student_predictions[0].studentId)
        self.db.add_all(student_predictions)
        self.db.commit()
        for prediction in student_predictions:
            self.db.refresh(prediction)
        logger.info("Batch student predictions saved successfully")
        return student_predictions
    
    def _convert_orm_to_dict(self, orm_objects, model_type):
        logger.debug(f"Converting {len(orm_objects)} {model_type} ORM objects to dict format")
        
        if model_type == 'course':
            result = []
            for obj in orm_objects:
                semester_name = obj.semester.name if hasattr(obj, 'semester') and obj.semester else ''
                course_dict = {
                    'Semester': semester_name,
                    'Course ID': obj.courseId,
                    'Course Name': obj.courseName,
                    'Credits': obj.credits,
                    'Class': obj.class_,
                    'Continuous Assessment Score': obj.continuousAssessmentScore or 0,
                    'Exam Score': obj.examScore or 0,
                    'Final Grade': obj.finalGrade or 'F',
                    'Relative Term': obj.relativeTerm,
                    'student_id': 1
                }
                result.append(course_dict)
            logger.debug(f"Converted {len(result)} course records")
            return result
            
        elif model_type == 'process':
            result = []
            for obj in orm_objects:
                semester_name = obj.semester.name if hasattr(obj, 'semester') and obj.semester else ''
                process_dict = {
                    'Semester': semester_name,
                    'student_id': 1,
                    'GPA': obj.gpa or 0,
                    'CPA': obj.cpa or 0,
                    'Relative Term': getattr(obj, 'studentLevel', 1),
                    'TC qua': obj.passedCredits or 0,
                    'Acc': obj.totalAcceptedCredits or 0,
                    'Debt': obj.debtCredits or 0,
                    'Reg': obj.registeredCredits or 0,
                    'Warning': f'Mức {obj.warningLevel}' if obj.warningLevel else 'Mức 0',
                    'Level': f'K{obj.studentLevel}' if hasattr(obj, 'studentLevel') else 'K1'
                }
                result.append(process_dict)
            logger.debug(f"Converted {len(result)} process records")
            return result
        
        logger.warning(f"Unknown model_type: {model_type}")
        return []
    
    def predict_for_all_students(self, current_semester_id: str, next_semester_id: str) -> List[StudentPrediction]:
        students = self.student_service.get_all_students_by_semester(
            semester_id=current_semester_id
        )
        predictions = []
        for student in students:
            prediction = self.predict_student_by_process(student.id, next_semester_id)
            predictions.extend(prediction)
        return predictions
    def predict_student_by_process(self, student_id: str, semester_id: str) -> List[StudentPrediction]:
        logger.info(f"Starting prediction for student: {student_id}, target semester: {semester_id}")
        
        try:
            logger.info("Fetching student data from database")
            student_info = self.student_service.get_student_info(student_id)
            print(student_info)
            student_courses = self.student_service.get_student_courses(student_id)
            student_process = self.student_service.get_student_process(student_id)
            
            if not student_info:
                logger.warning(f"Student info not found for ID: {student_id}")
                return []
            
            if not student_courses:
                logger.warning(f"No course records found for student: {student_id}")
                return []
                
            if not student_process:
                logger.warning(f"No process records found for student: {student_id}")
                return []
            
            logger.info(f"Retrieved data - Student: {student_info.name}, Courses: {len(student_courses)} records, Process: {len(student_process)} records")
            
            logger.info("Converting database records to ML model format")
            course_data = self._convert_orm_to_dict(student_courses, 'course')
            process_data = self._convert_orm_to_dict(student_process, 'process')
            
            logger.info(f"Data conversion completed - Course data: {len(course_data)} records, Process data: {len(process_data)} records")
            
            if process_data:
                latest_process = process_data[-1]
                logger.info(f"Latest student performance - GPA: {latest_process['GPA']}, CPA: {latest_process['CPA']}, Credits: {latest_process['TC qua']}")
            
            logger.info("Calling ML prediction service")
            prediction_result = self.prediction_service.predict_student_next_semester(
                course_data, process_data
            )
            
            if not prediction_result:
                logger.warning("ML prediction service returned no results")
                return []
            
            logger.info(f"ML prediction successful: GPA={prediction_result['predicted_gpa']:.4f}, CPA={prediction_result['predicted_cpa']:.4f}")
            
            logger.info("Calculating warning level based on predicted CPA")
            predicted_warning_level = 0
            if prediction_result['predicted_cpa'] < 1.0:
                predicted_warning_level = 3
                logger.info("Predicted warning level: 3 (Very High Risk)")
            elif prediction_result['predicted_cpa'] < 1.5:
                predicted_warning_level = 2
                logger.info("Predicted warning level: 2 (High Risk)")
            elif prediction_result['predicted_cpa'] < 2.0:
                predicted_warning_level = 1
                logger.info("Predicted warning level: 1 (Low Risk)")
            else:
                logger.info("Predicted warning level: 0 (No Risk)")
            
            logger.info("Creating StudentPrediction object")
            student_prediction = StudentPrediction(
                id=str(uuid.uuid4()),
                studentId=student_info.id,
                semesterId=semester_id,
                nextSemesterGPA=prediction_result['predicted_gpa'],
                nextSemesterCPA=prediction_result['predicted_cpa'],
                nextSemesterCredit=float(process_data[-1]['Reg']) if process_data else 15.0,
                nextSemesterWarningLevel=float(predicted_warning_level),
                createdAt=datetime.now(),
                updatedAt=datetime.now()
            )
            self.save_student_prediction(student_prediction)
            logger.info(f"StudentPrediction created successfully with ID: {student_prediction.id}")
            logger.info(f"Prediction summary: GPA={student_prediction.nextSemesterGPA:.4f}, CPA={student_prediction.nextSemesterCPA:.4f}, Credits={student_prediction.nextSemesterCredit}, Warning={student_prediction.nextSemesterWarningLevel}")
            
            return [student_prediction]
            
        except Exception as e:
            logger.error(f"Error in prediction for student {student_id}: {str(e)}", exc_info=True)
            return []
