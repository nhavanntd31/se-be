import { Injectable } from '@nestjs/common'
import { InjectRepository } from '@nestjs/typeorm'
import * as csv from 'csv-parse/sync'
import path from 'path'
import { Class } from 'src/database/entities/class'
import { Department } from 'src/database/entities/department'
import { Major } from 'src/database/entities/major'
import { Semester } from 'src/database/entities/semester'
import { Statistic } from 'src/database/entities/statistic'
import { QueueService } from 'src/services/queue/queue.service'
import { IsNull, Like, Repository, In } from 'typeorm'
import { GetStatisticInfoDto, GetCPATrajectoryDto, GetStudentsBySemesterRangeDto } from './dto'
import { ApiError } from 'src/common/responses'
import { Student } from 'src/database/entities/students'
import { UploadEvent } from 'src/database/entities/upload_event'
import { User } from 'src/database/entities/user.entity'
import { BaseSearchDto } from 'src/common/constants'
import { StudentCourse } from 'src/database/entities/student_course'
import { NotificationUser } from 'src/database/entities/notification_user'
import { Notification } from 'src/database/entities/notification'
import { StudentProcess } from 'src/database/entities/student_process'
import { generateSemesterList, isSemesterInRange } from 'src/common/utils/utils'
import { generateStudentPDFReport } from 'src/common/utils/llm_report.utils'
import * as fs from 'fs'
import * as crypto from 'crypto'
import { analyzePLOExcel } from 'src/common/utils/plo_analyze.utils'
import { ConfigService } from 'src/config/config.service'


@Injectable()
export class DataService {
  constructor(
    @InjectRepository(Semester) private semesterRepository: Repository<Semester>,
    @InjectRepository(Statistic) private statisticRepository: Repository<Statistic>,
    @InjectRepository(Department) private departmentRepository: Repository<Department>,
    @InjectRepository(Major) private majorRepository: Repository<Major>,
    @InjectRepository(Student) private studentRepository: Repository<Student>,
    @InjectRepository(Class) private classRepository: Repository<Class>,
    @InjectRepository(UploadEvent) private uploadEventRepository: Repository<UploadEvent>,
    @InjectRepository(StudentCourse) private studentCourseRepository: Repository<StudentCourse>,
    @InjectRepository(StudentProcess) private studentProcessRepository: Repository<StudentProcess>,
    @InjectRepository(Notification) private notificationRepository: Repository<Notification>,
    @InjectRepository(NotificationUser) private notiUserRepository: Repository<NotificationUser>,
    private queueService: QueueService,
    private configService: ConfigService
  ) {}

  async processCsv(file: Express.Multer.File) {

  }

  async uploadSemesterData(studentCourseCsv: Express.Multer.File, studentProcessCsv: Express.Multer.File, semester: string, user: User) {
    
    const uploadEvent = await this.uploadEventRepository.save({
      courseFilePath: studentCourseCsv.path,
      performanceFilePath: studentProcessCsv.path,
    })
    await this.queueService.addUpdateDatabaseQueue({
      studentCourseFilePath: studentCourseCsv.path,
      studentProcessFilePath: studentProcessCsv.path,
      uploadEvent,
      user
    })
    return true;
  }
  
  async runStatistic() {
    await this.queueService.addStatisticQueue({});
    return true;
  }
  async getSemester() {
    return await this.semesterRepository.find({
      order: {
        name: 'ASC'
      }
    })
  }
  async getDepartment() {
    return await this.departmentRepository.find({

    })
  }
  async getMajor() {
    return await this.majorRepository.find({
    })
  }
  async getClass() {
    return await this.classRepository.find({
    })
  }
  
  async getStatistic(body: GetStatisticInfoDto) {
    const semester = await this.semesterRepository.findOne({
      where: {
        id: body.semesterId
      }
    })
    
    if (!semester) {
      throw new ApiError('Semester not found');
    }
    if (body.departmentId) {
      const departmentStatistic = await this.statisticRepository.findOne({
        where: {
          departmentId: body.departmentId,
          semesterId: body.semesterId
        }
      })
      if (!departmentStatistic) {
        throw new ApiError('Department statistic not found');
      }
      
      return {
        ...departmentStatistic,
        averageCPA: await this.getAverageCPA(null, null, body.departmentId)
      };
    }
    if (body.majorId) {
      const majorStatistic = await this.statisticRepository.findOne({
        where: {
          majorId: body.majorId,
          semesterId: body.semesterId
        }
      })
      if (!majorStatistic) {
        throw new ApiError('Major statistic not found');
      }
      return {
        ...majorStatistic,
        averageCPA: await this.getAverageCPA(body.majorId, null, null)
      };
    }
    if (body.classId) {
      const classStatistic = await this.statisticRepository.findOne({
        where: {
          classId: body.classId,
          semesterId: body.semesterId
        }
      })
      if (!classStatistic) {  
        throw new ApiError('Class statistic not found');
      }
      return {
        ...classStatistic,
        averageCPA: await this.getAverageCPA(null, body.classId, null)
      };
    }
    const schoolStatistic = await this.statisticRepository.findOne({
      where: {
        semesterId: body.semesterId
      }
    })
    if (!schoolStatistic) {
      throw new ApiError('School statistic not found');
    }
    return {
      ...schoolStatistic,
      averageCPA: await this.getAverageCPA(null, null, null)
    };
  }

  async getAverageCPA(majorId?: string, classId?: string, departmentId?: string) {
    if (departmentId) {
      const departmentStatistic = await this.statisticRepository.find({
        select: {
          averageCPA: true,
          semester: {
            name: true
          }
        },
        where: {
          departmentId: departmentId
        },
        relations: ['semester'],
        order: {
          semester: {
            name: 'ASC'
          }
        }
      })
      if (!departmentStatistic) {
        throw new ApiError('Department statistic not found');
      }
      return departmentStatistic.map(statistic => ({
        averageCPA: statistic.averageCPA,
        semester: statistic.semester.name
      }));  
    }
    if (majorId) {
      const majorStatistic = await this.statisticRepository.find({
        select: {
          averageCPA: true,
          semester: {
            name: true
          }
        },
        where: {
          majorId: majorId
        },
        relations: ['semester'],
        order: {
          semester: {
            name: 'ASC'
          }
        }
      })
      if (!majorStatistic) {
        throw new ApiError('Major statistic not found');
      }
      return majorStatistic.map(statistic => ({
        averageCPA: statistic.averageCPA,
        semester: statistic.semester.name
      }));
    }
    if (classId) {
      const classStatistic = await this.statisticRepository.find({
        select: {
          averageCPA: true,
          semester: {
            name: true
          }
        },
        where: {
          classId: classId
        },
        relations: ['semester'],
        order: {
          semester: {
            name: 'ASC'
          }
        }
      })
      if (!classStatistic) {
        throw new ApiError('Class statistic not found');
      }
      return classStatistic.map(statistic => ({
        averageCPA: statistic.averageCPA,
        semester: statistic.semester.name
      }));
    }
    const schoolStatistic = await this.statisticRepository.find({
      select: {
        averageCPA: true,
        semester: {
          name: true
        }
      },
      where: {
        departmentId: IsNull(),
        majorId: IsNull(),
        classId: IsNull(),
      },
      relations: ['semester'],
      order: {
        semester: {
          name: 'ASC'
        }
      }
    })
    if (!schoolStatistic) {
      throw new ApiError('School statistic not found');
    }
    return schoolStatistic.map(statistic => ({
      averageCPA: statistic.averageCPA,
      semester: statistic.semester.name
    }));
  }

  async getStudentInfo(studentId: string) {
    const student = await this.studentRepository.findOne({
      select: {
        id: true,
        name: true,
        studentId: true,
        class: {
          name: true
        },
        major: {
          name: true
        },
        department: {
          name: true
        },
   
      },
      where: {
        studentId
      },
      relations: {
        department: true,
        major: true,
        class: true,
        studentProcesses: {
          semester: true
        },
        studentPredictions: {
          semester: true
        }
      },
      order: {
        studentProcesses: {
          semester: {
            name: 'DESC'
          }
        },
        studentPredictions: {
          semester: {
            name: 'DESC'
          }
        }
      }
    })
    if (!student) {
      throw new ApiError('Student not found');
    }
    return {
      ...student,
      studentProcesses: student.studentProcesses
        .sort((a, b) => a.semester.name.localeCompare(b.semester.name))
        .map(process => ({
        gpa: process.gpa,
        cpa: process.cpa,
        registeredCredits: process.registeredCredits,
        debtCredits: process.debtCredits,
        warningLevel: process.warningLevel,
        semester: process.semester.name,
        semesterId: process.semesterId
      }))
    }
  }
  
  async getStudentCourseBySemester(semesterId: string, studentId: string) {
    const studentCourse = await this.studentCourseRepository.find({
      where: {
        semesterId: semesterId,
        studentId: studentId
      }
    })
    return studentCourse
  }
  async getListUploadEvent(query: BaseSearchDto) {
    const page = query.offset || 1;
    const limit = query.limit || 10;

    const [uploadEvents, total] = await this.uploadEventRepository.findAndCount({
      skip: (page - 1) * limit,
      take: limit,
      order: {
        createdAt: 'DESC'
      }
    })

    return {
      data: uploadEvents,
      meta: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit)
      }
    }
  }

  async getUserNotification(query: BaseSearchDto, user: User) {
    const page = query.offset || 1;
    const limit = query.limit || 10;
    const [notiUser, total] = await this.notiUserRepository.findAndCount({
      where: {
        userId: user.id,
      },
      relations: {
        notification: true
      },
      skip: (page - 1) * limit,
      take: limit,
      order: {
        createdAt: 'DESC'
      }
    })
    return {
      data: notiUser.map(noti => noti.notification),
      meta: {
        page,
        limit,
        total,
        totalUnread: notiUser.filter(noti => !noti.isRead).length,
        totalPages: Math.ceil(total / limit)
      }
    }
  }
  async updateReadNotification(notiUserId: string, user: User) {
    const notiUser = await this.notiUserRepository.findOne({
      where: {
        id: notiUserId,
        userId: user.id
      }
    })
    if (!notiUser) {
      throw new ApiError('Notification not found');
    }
    notiUser.isRead = true
    await this.notiUserRepository.save(notiUser)
    return true
  }
  
  async getCPATrajectory(body: GetCPATrajectoryDto) {
    const startSemester = await this.semesterRepository.findOne({
      where: {
        id: body.startSemester
      }
    })
    const endSemester = await this.semesterRepository.findOne({
      where: {
        id: body.endSemester
      }
    })
    const semesterNames = generateSemesterList(startSemester.name, endSemester.name)
    
    const semesters = await this.semesterRepository.find({
      where: {
        name: In(semesterNames)
      },
      order: { name: 'ASC' }
    })

    const queryBuilder = this.studentRepository.createQueryBuilder('student')
      .leftJoinAndSelect('student.studentProcesses', 'process')
      .leftJoinAndSelect('process.semester', 'semester')
      .leftJoinAndSelect('student.class', 'class')
      .leftJoinAndSelect('student.major', 'major')
      .leftJoinAndSelect('student.department', 'department')
      .where('semester.name IN (:...semesterNames)', { semesterNames })

    if (body.departmentId) {
      queryBuilder.andWhere('student.departmentId = :departmentId', { departmentId: body.departmentId })
    }
    if (body.majorId) {
      queryBuilder.andWhere('student.majorId = :majorId', { majorId: body.majorId })
    }
    if (body.classId) {
      queryBuilder.andWhere('student.classId = :classId', { classId: body.classId })
    }
    queryBuilder.orderBy('semester.name', 'ASC')

    const students = await queryBuilder.getMany()
    console.log(students.length)
    const averageCPABySemester = []
    const studentDataBySemester = new Map()

    for (const semester of semesters) {
      const studentsInSemester = students.filter(student => 
        student.studentProcesses.some(process => process.semesterId === semester.id)
      )

      if (studentsInSemester.length === 0) continue

      const cpaValues = studentsInSemester.map(student => {
        const process = student.studentProcesses.find(p => p.semesterId === semester.id)
        return {
          studentId: student.studentId,
          cpa: process?.cpa || 0,
          studentName: student.name
        }
      }).filter(item => item.cpa > 0)

      if (cpaValues.length === 0) continue

      const averageCPA = cpaValues.reduce((sum, item) => sum + item.cpa, 0) / cpaValues.length

      averageCPABySemester.push({
        semester: semester.name,
        averageCPA: averageCPA
      })

      studentDataBySemester.set(semester.id, {
        semester: semester.name,
        students: cpaValues.sort((a, b) => Math.abs(a.cpa - averageCPA) - Math.abs(b.cpa - averageCPA))
      })
    }

    const allStudentTrajectories = []
    for (const student of students) {
      const sortedProcesses = student.studentProcesses.sort((a, b) => a.semester.name.localeCompare(b.semester.name))
      const trajectory = []
      for (const semester of semesters) {
        const process = sortedProcesses.find(p => p.semesterId === semester.id)
        if (process && process.cpa > 0) {
          trajectory.push({
            semester: semester.name,
            cpa: process.cpa
          })
        }
      }
      
      if (trajectory.length >= 2) {
        allStudentTrajectories.push({
          studentId: student.studentId,
          studentName: student.name,
          trajectory: trajectory
        })
      }
    }

    const averageTrajectoryMap = new Map()
    for (const avgPoint of averageCPABySemester) {
      averageTrajectoryMap.set(avgPoint.semester, avgPoint.averageCPA)
    }

    const calculateTrajectoryDistance = (studentTrajectory, averageTrajectory) => {
      let sumSquaredError = 0
      let validPoints = 0
      
      for (const point of studentTrajectory) {
        const avgCPA = averageTrajectory.get(point.semester)
        if (avgCPA !== undefined) {
          sumSquaredError += Math.pow(point.cpa - avgCPA, 2)
          validPoints++
        }
      }
      
      return validPoints > 0 ? sumSquaredError / validPoints : Infinity
    }

    const thresholdStudents = []
    for (const thresholdRate of body.thresholdRates) {
      const thresholdCount = Math.floor((allStudentTrajectories.length * thresholdRate) / 100)
      
      const studentsWithDistance = allStudentTrajectories.map(student => ({
        ...student,
        distanceToAverage: calculateTrajectoryDistance(student.trajectory, averageTrajectoryMap)
      }))
      
      const selectedStudents = studentsWithDistance
        .sort((a, b) => a.distanceToAverage - b.distanceToAverage)
        .slice(0, thresholdCount)

      const cpaTrajectories = selectedStudents.map(student => {
        const semesterCPAMap = new Map()
        student.trajectory.forEach(point => {
          semesterCPAMap.set(point.semester, point.cpa)
        })
        
        return semesters.map(semester => ({
          semester: semester.name,
          cpa: semesterCPAMap.get(semester.name) || 0
        }))
      })

      thresholdStudents.push({
        threshHold: thresholdRate,
        cpaTrajectory: cpaTrajectories
      })
    }

    const specificStudents = []
    if (body.studentIds && body.studentIds.length > 0) {
      for (const studentId of body.studentIds) {
        const student = students.find(s => s.studentId === studentId)
        if (student) {
          const semesterCPAMap = new Map()
          student.studentProcesses
            .sort((a, b) => a.semester.name.localeCompare(b.semester.name))
            .filter(process => process.cpa > 0)
            .forEach(process => {
              semesterCPAMap.set(process.semester.name, process.cpa)
            })
          
          const cpaTrajectory = semesters.map(semester => ({
            semester: semester.name,
            cpa: semesterCPAMap.get(semester.name) || 0
          }))
          
          specificStudents.push({
            studentId: student.studentId,
            studentName: student.name,
            cpaTrajectory: cpaTrajectory
          })
        }
      }
    }

    const medianCPABySemester = []
    for (const semester of semesters) {
      const studentsInSemester = students.filter(student => 
        student.studentProcesses.some(process => process.semesterId === semester.id)
      )

      if (studentsInSemester.length === 0) continue

      const cpaValues = studentsInSemester.map(student => {
        const process = student.studentProcesses.find(p => p.semesterId === semester.id)
        return process?.cpa || 0
      }).filter(cpa => cpa > 0).sort((a, b) => a - b)

      if (cpaValues.length === 0) continue

      const medianCPA = cpaValues.length % 2 === 0
        ? (cpaValues[cpaValues.length / 2 - 1] + cpaValues[cpaValues.length / 2]) / 2
        : cpaValues[Math.floor(cpaValues.length / 2)]

      medianCPABySemester.push({
        semester: semester.name,
        medianCPA: medianCPA
      })
    }

    const averageCPATrajectory = semesters.map(semester => {
      const avgData = averageCPABySemester.find(avg => avg.semester === semester.name)
      return {
        semester: semester.name,
        cpa: avgData ? avgData.averageCPA : 0
      }
    })

    const medianCPATrajectory = semesters.map(semester => {
      const medianData = medianCPABySemester.find(median => median.semester === semester.name)
      return {
        semester: semester.name,
        cpa: medianData ? medianData.medianCPA : 0
      }
    })

    return {
      averageCPA: averageCPATrajectory,
      medianCPA: medianCPATrajectory,
      thresholdStudents,
      specificStudents,
      semesters: semesters.map(s => s.name),
      totalStudents: students.length
    }
  }
  
  async getStudentsBySemesterRange(body: GetStudentsBySemesterRangeDto) {
    const page = body.offset || 1;
    const limit = body.limit || 10;

    const startSemester = await this.semesterRepository.findOne({
      where: {
        id: body.startSemester
      }
    })
    const endSemester = await this.semesterRepository.findOne({
      where: {
        id: body.endSemester
      }
    })
    
    const semesterNames = generateSemesterList(startSemester.name, endSemester.name)
    
    const semesters = await this.semesterRepository.find({
      where: {
        name: In(semesterNames)
      },
      order: { name: 'ASC' }
    })

    const queryBuilder = this.studentRepository.createQueryBuilder('student')
      .leftJoinAndSelect('student.studentProcesses', 'process')
      .leftJoinAndSelect('process.semester', 'semester')
      .leftJoinAndSelect('student.class', 'class')
      .leftJoinAndSelect('student.major', 'major')
      .leftJoinAndSelect('student.department', 'department')
      .where('semester.name IN (:...semesterNames)', { semesterNames })

    if (body.departmentId) {
      queryBuilder.andWhere('student.departmentId = :departmentId', { departmentId: body.departmentId })
    }
    if (body.majorId) {
      queryBuilder.andWhere('student.majorId = :majorId', { majorId: body.majorId })
    }
    if (body.classId) {
      queryBuilder.andWhere('student.classId = :classId', { classId: body.classId })
    }

    if (body.keyword) {
      queryBuilder.andWhere(
        '(student.studentId LIKE :keyword OR student.name LIKE :keyword)',
        { keyword: `%${body.keyword}%` }
      )
    }

    queryBuilder.orderBy('student.studentId', 'ASC')

    const [students, totalCount] = await queryBuilder.getManyAndCount()

    const studentsWithProcessData = students.map(student => ({
      studentId: student.studentId,
      studentName: student.name,
      class: student.class?.name,
      major: student.major?.name,
      department: student.department?.name,
      processes: student.studentProcesses
        .sort((a, b) => a.semester.name.localeCompare(b.semester.name))
        .filter(process => semesterNames.includes(process.semester.name))
        .map(process => ({
          semester: process.semester.name,
          gpa: process.gpa,
          cpa: process.cpa,
          registeredCredits: process.registeredCredits,
          debtCredits: process.debtCredits,
          warningLevel: process.warningLevel
        }))
    })).filter(student => student.processes.length > 0)

    const totalFilteredStudents = studentsWithProcessData.length
    const paginatedStudents = studentsWithProcessData.slice((page - 1) * limit, page * limit)

    return {
      data: paginatedStudents,
      meta: {
        page,
        limit,
        total: totalFilteredStudents,
        totalPages: Math.ceil(totalFilteredStudents / limit)
      },
      semesters: semesterNames,
      totalStudents: totalFilteredStudents
    }
  }

  async generateStudentPDFReport(studentId: string) {
    const student = await this.studentRepository.findOne({
      where: {
        id: studentId
      }
    })
    if (!student) {
      throw new ApiError('Student not found');
    }
    const studentProcesses = await this.studentProcessRepository.find({
      where: {
        studentId: studentId
      },
      relations: {
        semester: true
      },
      order: {
        semester: {
          name: 'ASC'
        }
      }
    })
    if (!studentProcesses) {
      throw new ApiError('Student processes not found');
    }
    const buffer = await generateStudentPDFReport(student, studentProcesses)
    return buffer
  }
  async analyzePLOExcel(files: { excel?: Express.Multer.File[], param?: Express.Multer.File[] }) {
    if (!files.excel?.length) throw new Error('Missing excel files')
    
    const paramBuffer = files.param?.[0] ? fs.readFileSync(files.param[0].path) : undefined
    const results = []
    
    for (let i = 0; i < files.excel.length; i++) {
      const file = files.excel[i]
      const excelBuffer = fs.readFileSync(file.path)
      
      try {
        const { analyzeBuffer, bloomBuffer, bloomTable } = await analyzePLOExcel(excelBuffer, paramBuffer, this.configService)
        
        results.push({
          fileIndex: i,
          fileName: file.originalname,
          analyze: analyzeBuffer.toString('base64'),
          bloom: bloomBuffer.toString('base64'),
          analyzeContent: analyzeBuffer.toString('utf-8'),
          bloomTable: bloomTable,
          analyzeContentType: 'text/markdown',
          bloomContentType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        })
      } catch (error) {
        results.push({
          fileIndex: i,
          fileName: file.originalname,
          error: error.message
        })
      }
    }
    
    return {
      results,
      totalFiles: files.excel.length,
      successfulFiles: results.filter(r => !r.error).length
    }
  }
}
