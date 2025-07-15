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
import { GetStatisticInfoDto, GetCPATrajectoryDto, GetStudentsBySemesterRangeDto, CLOSuggestDto, CLOCheckDto, PLOAnalyzeDto } from './dto';
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
    { name: 'excel', maxCount: 5 },
    { name: 'param', maxCount: 1 }
  ]))
  async analyzePLO(
    @UploadedFiles() files: { excel?: Express.Multer.File[], param?: Express.Multer.File[] },
    @Body() body: PLOAnalyzeDto
  ) {
    return await this.dataService.analyzePLOExcel(files, body);
  }

  @Post('suggest-clo')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'syllabus', maxCount: 1 },
    { name: 'param', maxCount: 1 }
  ]))
  async suggestCLO(
    @UploadedFiles() files: { syllabus?: Express.Multer.File[], param?: Express.Multer.File[] },
    @Body() body: CLOSuggestDto
  ) {
    return await this.dataService.suggestCLO(files, body);
  }

  @Post('check-clo')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'syllabus', maxCount: 1 },
    { name: 'clo', maxCount: 1 },
    { name: 'param', maxCount: 1 }
  ]))
  async checkCLO(
    @UploadedFiles() files: { syllabus?: Express.Multer.File[], clo?: Express.Multer.File[], param?: Express.Multer.File[] },
    @Body() body: CLOCheckDto
  ) {
    return await this.dataService.checkCLO(files, body);
  }

}
