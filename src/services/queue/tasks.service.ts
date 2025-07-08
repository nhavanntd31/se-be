import { MailerService } from '@nestjs-modules/mailer';
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Student } from 'src/database/entities/students';
import { StudentCourse } from 'src/database/entities/student_course';
import { StudentProcess } from 'src/database/entities/student_process';
import { Department } from 'src/database/entities/department';
import { In, IsNull, Repository } from 'typeorm';
import { Major } from 'src/database/entities/major';
import { Class } from 'src/database/entities/class';
import { StudentPrediction } from 'src/database/entities/students_predictions';
import { Semester } from 'src/database/entities/semester';
import { UploadEvent } from 'src/database/entities/upload_event';
import * as csv from 'csv-parse/sync';
import * as _ from 'lodash';
import { Utils } from 'src/common/utils/utils';
import { PythonUtils } from 'src/common/utils/python.utils';
import { Statistic } from 'src/database/entities/statistic';

@Injectable()
export class TasksService implements OnModuleInit {
  private readonly logger = new Logger(TasksService.name);

  constructor(
    private readonly mailerService: MailerService,
    @InjectRepository(Semester)
    private semesterRepository: Repository<Semester>,
    @InjectRepository(Student) private studentRepository: Repository<Student>,
    @InjectRepository(StudentProcess)
    private studentProcessRepository: Repository<StudentProcess>,
    @InjectRepository(StudentCourse)
    private studentCourseRepository: Repository<StudentCourse>,
    @InjectRepository(Department)
    private departmentRepository: Repository<Department>,
    @InjectRepository(Major) private majorRepository: Repository<Major>,
    @InjectRepository(Class) private classRepository: Repository<Class>,
    @InjectRepository(Statistic)
    private statisticRepository: Repository<Statistic>,
    @InjectRepository(UploadEvent)
    private uploadEventRepository: Repository<UploadEvent>,
  ) {}

  async sendMailProcess(job: any) {
    this.logger.log('Starting sendMailProcess');
    const data = job.data;
    await this.mailerService.sendMail(data);
    this.logger.log('Completed sendMailProcess');
  }

  async updateDatabaseProcess(job: any) {
    this.logger.log('Starting updateDatabaseProcess');
    const { studentCourseFilePath, studentProcessFilePath, uploadEvent } = job.data;
    
    this.logger.log('Processing student data...');
    await this.uploadEventRepository.update(uploadEvent.id, {
      importStartedAt: new Date(),
    });
    
    try {
      await this.processStudentData(
        studentProcessFilePath,
        studentCourseFilePath,
      );
      await this.uploadEventRepository.update(uploadEvent.id, {
        isImportSuccess: true,
        importCompletedAt: new Date(),
      });
      this.logger.log('Student data processing completed successfully');
    } catch (error) {
      this.logger.error(`Error processing student data: ${error.message}`);
      await this.uploadEventRepository.update(uploadEvent.id, {
        isImportSuccess: false,
        importCompletedAt: new Date(),
        importFailedMessage: error.message,
      });
    }

    this.logger.log('Running statistic process...');
    await this.uploadEventRepository.update(uploadEvent.id, {
      statisticStartedAt: new Date(),
    });
    
    try {
      await this.statisticProcess({});
      await this.uploadEventRepository.update(uploadEvent.id, {
        isStatisticSuccess: true,
        statisticCompletedAt: new Date(),
      });
      this.logger.log('Statistic process completed successfully');
    } catch (error) {
      this.logger.error(`Error running statistic process: ${error.message}`);
      await this.uploadEventRepository.update(uploadEvent.id, {
        isStatisticSuccess: false,
        statisticCompletedAt: new Date(),
        statisticFailedMessage: error.message,
      });
    }

    this.logger.log('Predicting student data...');
    await this.uploadEventRepository.update(uploadEvent.id, {
      predictStartedAt: new Date(),
    });
    
    try {
      await this.predictStudentData(
      );
      await this.uploadEventRepository.update(uploadEvent.id, {
        isPredictSuccess: true,
        predictCompletedAt: new Date(),
      });
      this.logger.log('Student data prediction completed successfully');
    } catch (error) {
      this.logger.error(`Error predicting student data: ${error.message}`);
      await this.uploadEventRepository.update(uploadEvent.id, {
        isPredictSuccess: false,
        predictCompletedAt: new Date(),
        predictFailedMessage: error.message,
      });
    }

    this.logger.log('Completed updateDatabaseProcess');
  }

  async processStudentData(
    studentProcessFilePath: string,
    studentCourseFilePath: string,
  ) {
    this.logger.log('Starting processStudentData');
    const fs = require('fs');
    this.logger.log('Reading student process file...');
    const studentProcessRaw = fs
      .readFileSync(studentProcessFilePath)
      .toString();
    const studentProcessContent = csv.parse(studentProcessRaw, {
      columns: true,
      skip_empty_lines: true,
    });

    this.logger.log('Reading student course file...');
    const studentCourseRaw = fs.readFileSync(studentCourseFilePath).toString();
    const studentCourseContent = csv.parse(studentCourseRaw, {
      columns: true,
      skip_empty_lines: true,
    });
    const department = await this.getDepartment('Khoa Điện tử');
    const semestersProcess = _.uniq(_.map(studentProcessContent, 'Semester'));
    const semestersCourse = _.uniq(_.map(studentCourseContent, 'Semester'));
    const majors = _.uniq(_.map(studentProcessContent, 'Prog'));

    this.logger.log('Creating semesters, majors and classes...');
    const [semesterEntitiesProcess, semesterEntitiesCourse, majorEntities] =
      await Promise.all([
        this.createSemesters(semestersProcess),
        this.createSemesters(semestersCourse),
        this.createMajors(majors, department),
      ]);
    const semesterEntities = [
      ...semesterEntitiesProcess,
      ...semesterEntitiesCourse,
    ];
    const classEntities = await this.createClasses(majorEntities);
    const students = _.uniq(_.map(studentProcessContent, 'student_id'));

    const studentRow = _.map(students, (studentId) => {
      const studentInfo = _.find(studentProcessContent, {
        student_id: studentId,
      });
      return {
        studentId,
        majorName: studentInfo.Prog,
      };
    });

    this.logger.log('Creating students...');
    const studentEntities = await this.createStudent(
      studentRow,
      department,
      majorEntities,
      classEntities,
    );

    this.logger.log('Creating student processes...');
    await this.createStudentProcess(
      studentProcessContent,
      semesterEntities,
      studentEntities,
    );

    this.logger.log('Creating student courses...');
    await this.createStudentCourse(
      studentCourseContent,
      studentEntities,
      semesterEntities,
    );

    this.logger.log('Completed processStudentData');
    return true;
  }
  private async predictStudentData(

  ) {
    this.logger.log('Starting predictStudentData');
    const semester = await this.semesterRepository.createQueryBuilder('semester')
      .innerJoin('semester.studentProcesses', 'studentProcess')
      .where('studentProcess.semesterId = semester.id')
      .getMany();
    const lasterSemester = this.getLastestSemesterByStringName(semester.map(semester => semester.name));
    const nextSemester = this.getRecentSemesterStringName(lasterSemester, false);
    let nextSemesterEntity = await this.semesterRepository.findOne({
      where: {
        name: nextSemester
      }
    });
    
    if (!nextSemesterEntity) {
      nextSemesterEntity = await this.semesterRepository.save({
        name: nextSemester,
      });
    }
    this.logger.log(`Predicting student for semester: ${lasterSemester} to ${nextSemester}`);
    try {
      const result = await PythonUtils.call('python/src/main.py', [
        semester.find(semester => semester.name === lasterSemester)?.id,
        nextSemesterEntity.id,
      ]);
    } catch (error) {
      this.logger.error(`Error predicting student: ${error.message}`);
      this.logger.error(`Full error details:`, error);
      throw error;
    }
    this.logger.log('Completed predictStudentData');
    return true;
  }

  private async getDepartment(name: string): Promise<Department> {
    this.logger.log(`Getting department: ${name}`);
    let department = await this.departmentRepository.findOne({
      where: { name },
    });
    if (!department) {
      this.logger.log(`Creating new department: ${name}`);
      department = new Department();
      department.name = name;
      return await this.departmentRepository.save(department);
    }
    return department;
  }

  private async createSemesters(semesters: string[]): Promise<Semester[]> {
    this.logger.log('Creating semesters');
    return Promise.all(
      _.map(semesters, async (semesterName) => {
        let semester = await this.semesterRepository.findOne({
          where: { name: semesterName },
        });
        if (!semester) {
          this.logger.log(`Creating new semester: ${semesterName}`);
          semester = new Semester();
          semester.name = semesterName;
          return await this.semesterRepository.upsert(semester, {
            conflictPaths: ['name'],
          });
        }
        return semester;
      }),
    );
  }

  private async createMajors(
    majors: string[],
    department: Department,
  ): Promise<Major[]> {
    this.logger.log('Creating majors');
    return Promise.all(
      _.map(majors, async (majorName) => {
        let major = await this.majorRepository.findOne({
          where: { name: majorName },
          relations: ['department'],
        });
        if (!major) {
          this.logger.log(`Creating new major: ${majorName}`);
          major = new Major();
          major.name = majorName;
          major.department = department;
          return await this.majorRepository.save(major);
        }
        return major;
      }),
    );
  }

  private async createClasses(majorEntities: Major[]): Promise<Class[]> {
    this.logger.log('Creating classes');
    return Promise.all(
      _.map(majorEntities, async (major) => {
        let classEntity = await this.classRepository.findOne({
          select: ['id', 'name'],
          where: { major: { id: major.id } },
          relations: ['major'],
        });
        if (!classEntity) {
          this.logger.log(`Creating new class for major: ${major.name}`);
          classEntity = new Class();
          classEntity.name = `${major.name} Class`;
          classEntity.major = major;
          return await this.classRepository.save(classEntity);
        }
        return classEntity;
      }),
    );
  }

  private async createStudent(
    studentRow: any[],
    department: Department,
    majorEntities: Major[],
    classEntities: Class[],
  ) {
    this.logger.log('Creating students');
    const existingStudents = await this.studentRepository.find({
      where: { studentId: In(_.map(studentRow, 'studentId')) },
    });
    const missingStudents = _.filter(
      studentRow,
      (student) => !_.find(existingStudents, ['studentId', student.studentId]),
    );

    if (missingStudents.length > 0) {
      this.logger.log(`Creating ${missingStudents.length} new students`);
    }

    const newStudents = missingStudents.map((student) => {
      const major = _.find(majorEntities, ['name', student.majorName]);
      const classEntity = _.find(classEntities, ['major.id', major.id]);
      return {
        studentId: student.studentId,
        name: student.studentId,
        majorId: major.id,
        classId: classEntity.id,
        departmentId: department.id,
      };
    });

    const savedNewStudents =
      newStudents.length > 0
        ? await this.studentRepository.save(newStudents)
        : [];
    return [...existingStudents, ...savedNewStudents];
  }

  private async createStudentProcess(
    studentProcessContent: any[],
    semesterEntities: Semester[],
    studentEntities: Student[],
  ) {
    this.logger.log('Creating student processes');
    const studentProcessEntities = _.map(studentProcessContent, (student) => {
      const studentEntity = _.find(studentEntities, [
        'studentId',
        String(student.student_id),
      ]);
      const semesterEntity = _.find(semesterEntities, [
        'name',
        student.Semester,
      ]);
      return {
        studentId: studentEntity.id,
        semesterId: semesterEntity.id,
        gpa: Utils.toFloat(student.GPA, 2),
        cpa: Utils.toFloat(student.CPA, 2),
        registeredCredits: parseInt(student.Reg),
        passedCredits: parseInt(student['TC qua']),
        debtCredits: parseInt(student.Debt),
        totalAcceptedCredits: parseInt(student.Acc),
        warningLevel: parseInt(student.Warning.split(' ')[1]) || 0,
        studentLevel: parseInt(student['Relative Term']),
      };
    });
    await this.studentProcessRepository.upsert(studentProcessEntities, {
      conflictPaths: ['studentId', 'semesterId'],
    });
  }

  private async createStudentCourse(
    studentCourseContent: any[],
    studentEntities: Student[],
    semesterEntities: Semester[],
  ) {
    this.logger.log('Creating student courses');
    const studentCourseEntities = studentCourseContent.map((row) => {
      const studentEntity = studentEntities.find(
        (s) => s.studentId === String(row.student_id),
      );
      const semesterEntity = semesterEntities.find(
        (s) => s.name === row.Semester,
      );
      return {
        studentId: studentEntity.id,
        semesterId: semesterEntity.id,
        courseId: row['Course ID'],
        courseName: row['Course Name'],
        credits: parseInt(row['Credits']),
        class: row['Class'],
        continuousAssessmentScore: Utils.toFloat(
          row['Continuous Assessment Score'],
          2,
        ),
        examScore: Utils.toFloat(row['Exam Score'], 2),
        finalGrade: row['Final Grade'],
        relativeTerm: parseInt(row['Relative Term']),
      };
    });

    const batchSize = 1000;
    const batches = [];
    for (let i = 0; i < studentCourseEntities.length; i += batchSize) {
      batches.push(studentCourseEntities.slice(i, i + batchSize));
    }
    
    this.logger.log(`Processing ${batches.length} batches of ${batchSize} records each`);
    
    const result = await Promise.all(
      batches.map(async (batch, index) => {
        this.logger.log(`Processing batch ${index + 1}/${batches.length}`);
        return this.studentCourseRepository.upsert(batch, {
          conflictPaths: ['studentId', 'semesterId', 'courseId'],
        });
      })
    );
    return true;
  }

  async statisticProcess(job: any) {
    this.logger.log('Statistic process started');
    const listSemester = await this.semesterRepository.find();
    const listDepartment = await this.departmentRepository.find();
    const listMajor = await this.majorRepository.find();
    const listClass = await this.classRepository.find();
    
    for (const semester of listSemester) {
      this.logger.log(`Processing statistics for semester: ${semester.name}`);
      await this.statisticDepartment(semester.id, listDepartment);
      await this.statisticMajor(semester.id, listMajor);
      await this.statisticClass(semester.id, listClass);
      await this.statisticSchool(semester.id);
    }
    this.logger.log('Statistic process completed');
    return true;
  }

  async statisticDepartment(semesterId: string, listDepartment: Department[]) {
    this.logger.log('Processing department statistics');
    const result = await Promise.all(listDepartment.map(async (department) => {
      const semester = await this.semesterRepository.findOne({
        where: {
          id: semesterId
        }
      });
      const listStudent = await this.studentRepository.find({
        where: {
          department: {
            id: department.id
          }
        }
      });
      const listStudentProcess = await this.studentProcessRepository.find({
        where: {
          student: {
            department: {
              id: department.id
            }
          },
          semester: {
            id: semesterId
          }
        }
      });
      const result = await this.getStudentStatistic(semester.name, listStudentProcess, []);
      await this.statisticRepository.upsert({
        name: department.name,
        semester: semester,
        department: department,
        averageCPA: parseFloat(result.averageCPA),
        averageGPA: parseFloat(result.averageGPA),
        totalStudents: result.totalStudents,
        totalStudentIn: result.totalStudentIn,
        totalStudentOut: result.totalStudentOut,
        studentGraduationOnTimeRate: result.studentGraduationOnTimeRate,
        studentGraduationLateNumber: result.studentGraduationLateNumber,
        studentGraduationLateRate: result.studentGraduationLateRate,
        studentUngraduationNumber: result.studentUndergraduationNumber,
        studentUngraduationRate: result.studentUndergraduationRate,
        studentInWarningNumber: result.studentInWarningNumber,
        studentInWarningRate: result.studentInWarningRate,
        studentWarningOneRate: result.studentWarningOneRate,
        studentWarningTwoRate: result.studentWarningTwoRate,
        studentWarningThreeRate: result.studentWarningThreeRate,
        studentExcellentRate: result.studentExcellentRate,
        studentVeryGoodRate: result.studentVeryGoodRate,
        studentGoodRate: result.studentGoodRate,
        studentMediumRate: result.studentMediumRate,
        studentBadRate: result.studentBadRate,
      }, {
        conflictPaths: ['semesterId', 'departmentId'],
      });
    }))
  }


  async statisticSchool(semesterId: string) {
    this.logger.log('Processing school statistics');
    const semester = await this.semesterRepository.findOne({
      where: {
        id: semesterId
      }
    });
    const listStudentProcess = await this.studentProcessRepository.find({
      where: {
        semester: {
          id: semesterId
        }
      }
    });
    const result = await this.getStudentStatistic(semester.name, listStudentProcess, []);
    await this.statisticRepository.upsert({
      semester: semester,
      averageCPA: parseFloat(result.averageCPA),
      averageGPA: parseFloat(result.averageGPA),
      totalStudents: result.totalStudents,
      totalStudentIn: result.totalStudentIn,
      totalStudentOut: result.totalStudentOut,
      studentGraduationOnTimeRate: result.studentGraduationOnTimeRate,
      studentGraduationLateNumber: result.studentGraduationLateNumber,
      studentGraduationLateRate: result.studentGraduationLateRate,
      studentUngraduationNumber: result.studentUndergraduationNumber,
      studentUngraduationRate: result.studentUndergraduationRate,
      studentInWarningNumber: result.studentInWarningNumber,
      studentInWarningRate: result.studentInWarningRate,
      studentWarningOneRate: result.studentWarningOneRate,
      studentWarningTwoRate: result.studentWarningTwoRate,
      studentWarningThreeRate: result.studentWarningThreeRate,
      studentExcellentRate: result.studentExcellentRate,
      studentVeryGoodRate: result.studentVeryGoodRate,
      studentGoodRate: result.studentGoodRate,
      studentMediumRate: result.studentMediumRate,
      studentBadRate: result.studentBadRate,
    }, {
      conflictPaths: ['name', 'semester', 'major'],
    });
  }
  async statisticMajor(semesterId: string, listMajor: Major[]) {
    this.logger.log('Processing major statistics');
    const result = await Promise.all(listMajor.map(async (major) => {
      const semester = await this.semesterRepository.findOne({
        where: {
          id: semesterId
        }
      });
      const listStudentProcess = await this.studentProcessRepository.find({
        where: {
          student: {
            major: {
              id: major.id
            }
          },
          semester: {
            id: semesterId
          }
        }
      });
      const result = await this.getStudentStatistic(semester.name, listStudentProcess, []);
      await this.statisticRepository.upsert({
        name: major.name,
        semester: semester,
        major: major,
        averageCPA: parseFloat(result.averageCPA),
        averageGPA: parseFloat(result.averageGPA),
        totalStudents: result.totalStudents,
        totalStudentIn: result.totalStudentIn,
        totalStudentOut: result.totalStudentOut,
        studentGraduationOnTimeRate: result.studentGraduationOnTimeRate,
        studentGraduationLateNumber: result.studentGraduationLateNumber,
        studentGraduationLateRate: result.studentGraduationLateRate,
        studentUngraduationNumber: result.studentUndergraduationNumber,
        studentUngraduationRate: result.studentUndergraduationRate,
        studentInWarningNumber: result.studentInWarningNumber,
        studentInWarningRate: result.studentInWarningRate,
        studentWarningOneRate: result.studentWarningOneRate,
        studentWarningTwoRate: result.studentWarningTwoRate,
        studentWarningThreeRate: result.studentWarningThreeRate,
        studentExcellentRate: result.studentExcellentRate,
        studentVeryGoodRate: result.studentVeryGoodRate,
        studentGoodRate: result.studentGoodRate,
        studentMediumRate: result.studentMediumRate,
        studentBadRate: result.studentBadRate,
      }, {
        conflictPaths: ['semesterId', 'majorId'],
      });
    }));
  }

  async statisticClass(semesterId: string, listClass: Class[]) {
    this.logger.log('Processing class statistics');
    const result = await Promise.all(listClass.map(async (cls) => {
      const semester = await this.semesterRepository.findOne({
        where: {
          id: semesterId
        }
      });
      const listStudent = await this.studentRepository.find({
        where: {
          class: {
            id: cls.id
          }
        }
      });
      const listStudentProcess = await this.studentProcessRepository.find({
        where: {
          student: {
            class: {
              id: cls.id
            }
          },
          semester: {
            id: semesterId
          }
        }
      });
      const result = await this.getStudentStatistic(semester.name, listStudentProcess, []);  
      await this.statisticRepository.upsert({
        name: cls.name,
        semester: semester,
        class: cls,
        averageCPA: parseFloat(result.averageCPA),
        averageGPA: parseFloat(result.averageGPA),
        totalStudents: result.totalStudents,
        totalStudentIn: result.totalStudentIn,
        totalStudentOut: result.totalStudentOut,
        studentGraduationOnTimeRate: result.studentGraduationOnTimeRate,
        studentGraduationLateRate: result.studentGraduationLateRate,
        studentGraduationLateNumber: result.studentGraduationLateNumber,
        studentUngraduationNumber: result.studentUndergraduationNumber,
        studentUngraduationRate: result.studentUndergraduationRate,
        studentInWarningNumber: result.studentInWarningNumber,
        studentInWarningRate: result.studentInWarningRate,
        studentWarningOneRate: result.studentWarningOneRate,
        studentWarningTwoRate: result.studentWarningTwoRate,
        studentWarningThreeRate: result.studentWarningThreeRate,
        studentExcellentRate: result.studentExcellentRate,
        studentVeryGoodRate: result.studentVeryGoodRate,
        studentGoodRate: result.studentGoodRate,
        studentMediumRate: result.studentMediumRate,
        studentBadRate: result.studentBadRate,
      }, {
        conflictPaths: ['semesterId', 'classId'],
      });
    }))
  } 
  async getStudentStatistic(semesterName: string, listStudentProcess: StudentProcess[], listStudentCourse: StudentCourse[]) {
    this.logger.log(`Getting student statistics for semester: ${semesterName}`);
    const totalStudents = _.uniq(listStudentProcess.map(process => process.studentId)).length;
    const totalStudentIn = listStudentProcess.filter(studentProcess => studentProcess.studentLevel === 1).length;
    const previousSemesterName = this.getRecentSemesterStringName(semesterName, true);
    const previousSemesterStudentProcess = await this.studentProcessRepository.find({
      where: {
        semester: {
          name: previousSemesterName
        }
      }
    });
    const totalStudentOut = previousSemesterStudentProcess.filter(studentProcess => 
      studentProcess.totalAcceptedCredits > 125 && 
      !listStudentProcess.find(sp => sp.studentId === studentProcess.studentId)
    ).length;
    
    const totalGraduatedOnTime = previousSemesterStudentProcess.filter(studentProcess => 
      studentProcess.totalAcceptedCredits > 125 && 
      studentProcess.studentLevel < 10 &&
      !listStudentProcess.find(sp => sp.studentId === studentProcess.studentId)
    ).length;

    const studentGraduationOnTimeRate = totalStudentIn > 0 ? (totalGraduatedOnTime / totalStudentIn * 100) : 0;
    const studentGraduationLateNumber = totalStudentOut - totalGraduatedOnTime;
    const studentGraduationLateRate = totalStudentIn > 0 ? (studentGraduationLateNumber / totalStudentIn * 100) : 0;
    const studentUndergraduationNumber = listStudentProcess.filter(studentProcess => 
      previousSemesterStudentProcess.find(sp => sp.studentId === studentProcess.studentId)
    ).length;
    const studentUndergraduationRate = totalStudents > 0 ? (studentUndergraduationNumber / totalStudents * 100) : 0;
    const studentInWarningNumber = listStudentProcess.filter(studentProcess => 
      studentProcess.warningLevel > 0    ).length;
    const studentInWarningRate = totalStudents > 0 ? (studentInWarningNumber / totalStudents * 100) : 0;
    const studentWarningOne = listStudentProcess.filter(studentProcess => 
      studentProcess.warningLevel === 1
    ).length;
    const studentWarningTwo= listStudentProcess.filter(studentProcess => 
      studentProcess.warningLevel === 2
    ).length;
    const studentWarningThree = listStudentProcess.filter(studentProcess => 
      studentProcess.warningLevel === 3
    ).length;
    const studentWarningOneRate = studentInWarningNumber > 0 ? (studentWarningOne / studentInWarningNumber * 100) : 0;
    const studentWarningTwoRate = studentInWarningNumber > 0 ? (studentWarningTwo / studentInWarningNumber * 100) : 0;
    const studentWarningThreeRate = studentInWarningNumber > 0 ? (studentWarningThree / studentInWarningNumber * 100) : 0;
    const studentExcellent = listStudentProcess.filter(studentProcess => 
      studentProcess.cpa >= 3.6
    ).length;
    const studentVeryGood = listStudentProcess.filter(studentProcess => 
      studentProcess.cpa >= 3.2 && studentProcess.cpa < 3.6
    ).length;
    const studentGood = listStudentProcess.filter(studentProcess => 
      studentProcess.cpa >= 2.5 && studentProcess.cpa < 3.2
    ).length;
    const studentMedium = listStudentProcess.filter(studentProcess => 
      studentProcess.cpa >= 2.0 && studentProcess.cpa < 2.5
    ).length;
    const studentBad = listStudentProcess.filter(studentProcess => 
      studentProcess.cpa < 2.0
    ).length;
    const studentExcellentRate = totalStudents > 0 ? (studentExcellent / totalStudents * 100) : 0;
    const studentVeryGoodRate = totalStudents > 0 ? (studentVeryGood / totalStudents * 100) : 0;
    const studentGoodRate = totalStudents > 0 ? (studentGood / totalStudents * 100) : 0;
    const studentMediumRate = totalStudents > 0 ? (studentMedium / totalStudents * 100) : 0;
    const studentBadRate = totalStudents > 0 ? (studentBad / totalStudents * 100) : 0;

    const averageCPA = listStudentProcess.length > 0 ? 
      (listStudentProcess.reduce((acc, studentProcess) => acc + studentProcess.cpa, 0) / listStudentProcess.length) : 0;
    const averageGPA = listStudentProcess.length > 0 ? 
      (listStudentProcess.reduce((acc, studentProcess) => acc + studentProcess.gpa, 0) / listStudentProcess.length) : 0;
    return {
      totalStudents,
      totalStudentIn,
      totalStudentOut,
      studentGraduationOnTimeRate,
      studentGraduationLateNumber,
      studentGraduationLateRate,
      studentUndergraduationNumber,
      studentUndergraduationRate,
      studentInWarningNumber,
      studentInWarningRate,
      studentWarningOneRate,
      studentWarningTwoRate,
      studentWarningThreeRate,
      studentExcellentRate,
      studentVeryGoodRate,
      studentGoodRate,
      studentMediumRate,
      studentBadRate,
      averageCPA: averageCPA.toFixed(2),
      averageGPA: averageGPA.toFixed(2),
    }
  }

  getRecentSemesterStringName(semesterName: string, isPrevious: boolean) {
    const year = parseInt(semesterName.substring(0, 4));
    const term = parseInt(semesterName.substring(4));
    
    if (isPrevious) {
      if (term === 1) {
        return `${year - 1}3`;
      }
      return `${year}${term - 1}`;
    } else {
      if (term === 3) {
        return `${year + 1}1`;
      }
      return `${year}${term + 1}`;
    }
  }
  getLastestSemesterByStringName(semesterNames: string[]) {
    let latestSemester = semesterNames[0];
    
    for (const semester of semesterNames) {
      const currentYear = parseInt(semester.substring(0, 4));
      const currentTerm = parseInt(semester.substring(4));
      
      const latestYear = parseInt(latestSemester.substring(0, 4));
      const latestTerm = parseInt(latestSemester.substring(4));

      if (currentYear > latestYear || (currentYear === latestYear && currentTerm > latestTerm)) {
        latestSemester = semester;
      }
    }

    return latestSemester;
  }

  async onModuleInit() {}
}
