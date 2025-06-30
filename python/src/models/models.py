from sqlalchemy import Column, String, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship
from models.base import BaseModel
import uuid

class Department(BaseModel):
    __tablename__ = 'departments'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    students = relationship('Student', back_populates='department')
    majors = relationship('Major', back_populates='department')

class Major(BaseModel):
    __tablename__ = 'majors'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    departmentId = Column(String, ForeignKey('departments.id'))
    department = relationship('Department', back_populates='majors')
    students = relationship('Student', back_populates='major')
    classes = relationship('Class', back_populates='major')

class Class(BaseModel):
    __tablename__ = 'classes'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    majorId = Column(String, ForeignKey('majors.id'))
    major = relationship('Major', back_populates='classes')
    students = relationship('Student', back_populates='class_')

class Student(BaseModel):
    __tablename__ = 'students'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    studentId = Column(String(256), unique=True)
    majorId = Column(String, ForeignKey('majors.id'))
    classId = Column(String, ForeignKey('classes.id'))
    departmentId = Column(String, ForeignKey('departments.id'))
    
    major = relationship('Major', back_populates='students')
    class_ = relationship('Class', back_populates='students')
    department = relationship('Department', back_populates='students')
    student_courses = relationship('StudentCourse', back_populates='student')
    student_processes = relationship('StudentProcess', back_populates='student')
    student_predictions = relationship('StudentPrediction', back_populates='student')

class Semester(BaseModel):
    __tablename__ = 'semesters'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    student_courses = relationship('StudentCourse', back_populates='semester')
    student_processes = relationship('StudentProcess', back_populates='semester')
    student_predictions = relationship('StudentPrediction', back_populates='semester')

class StudentCourse(BaseModel):
    __tablename__ = 'student_courses'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    studentId = Column(String, ForeignKey('students.id'))
    semesterId = Column(String, ForeignKey('semesters.id'))
    courseId = Column(String)
    courseName = Column(String(256))
    credits = Column(Integer)
    class_ = Column('class', String(256))
    continuousAssessmentScore = Column(Float)
    examScore = Column(Float)
    finalGrade = Column(String(10))
    relativeTerm = Column(Integer)
    
    student = relationship('Student', back_populates='student_courses')
    semester = relationship('Semester', back_populates='student_courses')

class StudentProcess(BaseModel):
    __tablename__ = 'student_processes'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    studentId = Column(String, ForeignKey('students.id'))
    semesterId = Column(String, ForeignKey('semesters.id'))
    gpa = Column(Float)
    cpa = Column(Float)
    registeredCredits = Column(Integer)
    passedCredits = Column(Integer)
    debtCredits = Column(Integer)
    totalAcceptedCredits = Column(Integer)
    warningLevel = Column(Integer)
    studentLevel = Column(Integer)
    
    student = relationship('Student', back_populates='student_processes')
    semester = relationship('Semester', back_populates='student_processes')

class StudentPrediction(BaseModel):
    __tablename__ = 'students_predictions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    studentId = Column(String, ForeignKey('students.id'))
    semesterId = Column(String, ForeignKey('semesters.id'))
    nextSemesterGPA = Column(Float)
    nextSemesterCredit = Column(Float)
    nextSemesterCPA = Column(Float)
    nextSemesterWarningLevel = Column(Float)

    student = relationship('Student', back_populates='student_predictions')
    semester = relationship('Semester', back_populates='student_predictions')