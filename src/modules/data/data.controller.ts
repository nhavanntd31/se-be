import {
  Controller,
  Post,
  Get,
  Query,
  UploadedFiles,
  UseInterceptors,
  Param,
  UseGuards,
  Put,
  Body,
} from '@nestjs/common';
import {
  FileFieldsInterceptor,
  FilesInterceptor,
} from '@nestjs/platform-express';
import { DataService } from './data.service';
import { GetStatisticInfoDto, GetCPATrajectoryDto, GetStudentsBySemesterRangeDto } from './dto';
import { CurrentUser } from 'src/common/decorators/user.decorator';
import { JwtAuthGuard } from 'src/common/guards/jwt-auth.guard';
import { ApiBearerAuth, ApiUnauthorizedResponse } from '@nestjs/swagger';
import { User, UserRole } from 'src/database/entities';
import { Roles } from 'src/common/decorators/role.decorator';
import { PublicGuard, RolesGuard } from 'src/common/guards';
import { BaseSearchDto } from 'src/common/constants';
import { analyzePLOExcel } from 'src/common/utils/plo_analyze.utils'
import * as fs from 'fs'
import axios from 'axios'
import FormData from 'form-data'

@Controller('data')
@UseGuards(JwtAuthGuard, PublicGuard)
@ApiBearerAuth()
@ApiUnauthorizedResponse({ description: 'Unauthorized' })
export class DataController {
  constructor(private readonly dataService: DataService) {}

  @Post('upload-csv')
  @UseInterceptors(
    FileFieldsInterceptor([
      { name: 'studentCourseCsv', maxCount: 1 },
      { name: 'studentProcessCsv', maxCount: 1 },
    ]),
  )

  uploadCsv(
    @UploadedFiles()
    files: {
      studentCourseCsv: Express.Multer.File[];
      studentProcessCsv: Express.Multer.File[];
    },
    @Query('semester') semester: string,
    @CurrentUser() user: User
  ) {
    const studentCourseCsv = files.studentCourseCsv[0];
    const studentProcessCsv = files.studentProcessCsv[0];
    return this.dataService.uploadSemesterData(
      studentCourseCsv,
      studentProcessCsv,
      semester,
      user,
    );
  }

  @Get('run-statistic')
  runStatistic() {
    return this.dataService.runStatistic();
  }

  @Get('statistic')
  getStatistic(@Query() body: GetStatisticInfoDto) {
    return this.dataService.getStatistic(body);
  }

  @Get('department')
  async getDepartment() {
    return await this.dataService.getDepartment();
  }

  @Get('major')
  async getMajor() {
    return await this.dataService.getMajor();
  }

  @Get('class')
  async getClass() {
    return await this.dataService.getClass();
  }

  @Get('semester')
  async getSemester() {
    return await this.dataService.getSemester();
  }
  @Get('student/:id')
  async getStudentInfo(@Param('id') id: string) {
    return await this.dataService.getStudentInfo(id);
  }
  @Get('student-course')
  async getStudentCourseBySemester(@Query('semesterId') semesterId: string, @Query('studentId') studentId: string) {
    return await this.dataService.getStudentCourseBySemester(semesterId, studentId);
}
  @Get('upload-event')
  async getListUploadEvent(@Query() query: BaseSearchDto) {
    return await this.dataService.getListUploadEvent(query);
  }

  @Get('notification')
  async listNotification(@Query() query: BaseSearchDto, @CurrentUser() user: User) {
    return await this.dataService.getUserNotification(query, user);
  }

  @Put('notification/:id')
  async updateReadNotification(@Param('id') id: string, @CurrentUser() user: User) {
    return await this.dataService.updateReadNotification(id, user);
  }

  @Post('cpa-trajectory')
  async getCPATrajectory(@Body() body: GetCPATrajectoryDto) {
    return await this.dataService.getCPATrajectory(body);
  }

  @Get('students-by-semester-range')
  async getStudentsBySemesterRange(@Query() query: GetStudentsBySemesterRangeDto) {
    return await this.dataService.getStudentsBySemesterRange(query);
  }

  @Get('generate-student-pdf-report/:studentId')
  async generateStudentPDFReport(@Param('studentId') studentId: string) {
    const buffer = await this.dataService.generateStudentPDFReport(studentId);
    return {
      buffer: buffer.toString('base64'),
      contentType: 'application/pdf'
    }
  }

  @Post('analyze-plo')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'excel', maxCount: 1 },
    { name: 'param', maxCount: 1 }
  ]))
  async analyzePLO(@UploadedFiles() files: { excel?: Express.Multer.File[], param?: Express.Multer.File[] }) {
    if (!files.excel?.[0]) throw new Error('Missing excel file')
    const excelBuffer = fs.readFileSync(files.excel[0].path)
    const paramBuffer = files.param?.[0] ? fs.readFileSync(files.param[0].path) : undefined
    const { analyzeBuffer, bloomBuffer, bloomTable } = await analyzePLOExcel(excelBuffer, paramBuffer)
    return {
      analyze: analyzeBuffer.toString('base64'),
      bloom: bloomBuffer.toString('base64'),
      analyzeContent: analyzeBuffer.toString('utf-8'),
      bloomTable: bloomTable,
      analyzeContentType: 'text/markdown',
      bloomContentType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
  }

  @Post('predict-students')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'course', maxCount: 1 },
    { name: 'performance', maxCount: 1 }
  ]))
  async predictStudents(@UploadedFiles() files: { course?: Express.Multer.File[], performance?: Express.Multer.File[] }) {
    if (!files.course?.[0] || !files.performance?.[0]) {
      throw new Error('Missing course or performance file')
    }

    try {
      const courseBuffer = fs.readFileSync(files.course[0].path)
      const performanceBuffer = fs.readFileSync(files.performance[0].path)

      const formData = new FormData()
      formData.append('course_file', courseBuffer, files.course[0].originalname)
      formData.append('performance_file', performanceBuffer, files.performance[0].originalname)

      const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:8001'
      const response = await axios.post(`${pythonApiUrl}/predict-students`, formData, {
        headers: formData.getHeaders(),
        timeout: 300000
      })

      return response.data
    } catch (error) {
      throw new Error(`Prediction service error: ${error.message}`)
    }
  }

  @Post('predict-student-scenarios')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'course', maxCount: 1 },
    { name: 'performance', maxCount: 1 }
  ]))
  async predictStudentScenarios(@UploadedFiles() files: { course?: Express.Multer.File[], performance?: Express.Multer.File[] }) {
    if (!files.course?.[0] || !files.performance?.[0]) {
      throw new Error('Missing course or performance file')
    }

    try {
      const courseBuffer = fs.readFileSync(files.course[0].path)
      const performanceBuffer = fs.readFileSync(files.performance[0].path)

      const formData = new FormData()
      formData.append('course_file', courseBuffer, files.course[0].originalname)
      formData.append('performance_file', performanceBuffer, files.performance[0].originalname)

      const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:8001'
      const response = await axios.post(`${pythonApiUrl}/predict-student-scenarios`, formData, {
        headers: formData.getHeaders(),
        timeout: 300000
      })

      return response.data
    } catch (error) {
      throw new Error(`Scenario prediction service error: ${error.message}`)
    }
  }
}
