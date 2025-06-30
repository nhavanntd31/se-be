from sqlalchemy.orm import Session
from sqlalchemy import and_
from models.models import Student, StudentProcess, StudentCourse, Department, Major, Class, Semester
from typing import List, Optional

class StudentService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_student_by_id(self, student_id: str) -> Optional[Student]:
        return self.db.query(Student).filter(Student.id == student_id).first()
    
    def get_student_info(self, student_id: str) -> Optional[Student]:
        return self.get_student_by_id(student_id)
    
    def get_student_process(self, student_id: str) -> List[StudentProcess]:
        return self.get_student_processes(student_id)
    
    def get_all_students(self) -> List[Student]:
        return self.db.query(Student).filter(Student.isDeleted == False).all()
    def get_all_students_by_semester(self, semester_id: str) -> List[Student]:
        return self.db.query(Student).filter(
            Student.student_processes.any(
                and_(
                    StudentProcess.semester.has(id=semester_id),
                    StudentProcess.isDeleted == False
                )
            ),
            Student.isDeleted == False
        ).all()
    def create_student(self, student_data: dict) -> Student:
        student = Student(**student_data)
        self.db.add(student)
        self.db.commit()
        self.db.refresh(student)
        return student
    
    def update_student(self, student_id: str, student_data: dict) -> Optional[Student]:
        student = self.get_student_by_id(student_id)
        if student:
            for key, value in student_data.items():
                setattr(student, key, value)
            self.db.commit()
            self.db.refresh(student)
        return student
    
    def delete_student(self, student_id: str) -> bool:
        student = self.get_student_by_id(student_id)
        if student:
            student.isDeleted = True
            self.db.commit()
            return True
        return False
    
    def get_student_processes(self, student_id: str) -> List[StudentProcess]:
        return self.db.query(StudentProcess).filter(
            StudentProcess.studentId == student_id,
            StudentProcess.isDeleted == False
        ).all()
    
    def get_student_courses(self, student_id: str) -> List[StudentCourse]:
        return self.db.query(StudentCourse).filter(
            StudentCourse.studentId == student_id,
            StudentCourse.isDeleted == False
        ).all()
    
    def get_department_by_name(self, name: str) -> Optional[Department]:
        return self.db.query(Department).filter(Department.name == name).first()
    
    def get_major_by_name(self, name: str) -> Optional[Major]:
        return self.db.query(Major).filter(Major.name == name).first()
    
    def get_class_by_name(self, name: str) -> Optional[Class]:
        return self.db.query(Class).filter(Class.name == name).first()
    
    def get_semester_by_name(self, name: str) -> Optional[Semester]:
        return self.db.query(Semester).filter(Semester.name == name).first()