import { Module } from '@nestjs/common'
import { TypeOrmModule } from '@nestjs/typeorm'
import { DataController } from './data.controller'
import { DataService } from './data.service'
import { Semester } from 'src/database/entities/semester'
import { Department } from 'src/database/entities/department'
import { Major } from 'src/database/entities/major'
import { Class } from 'src/database/entities/class'
import { Student } from 'src/database/entities/students'
import { StudentProcess } from 'src/database/entities/student_process'
import { StudentCourse } from 'src/database/entities/student_course'
import { QueueModule } from 'src/services/queue/queue.module'
import { MulterModule } from '@nestjs/platform-express'
import { Statistic } from 'src/database/entities/statistic'
import { UploadEvent } from 'src/database/entities/upload_event'
import { NotificationUser } from 'src/database/entities/notification_user'
import { Notification } from 'src/database/entities/notification'
import { ConfigModule } from 'src/config/config.module'
@Module({
  imports: [
    TypeOrmModule.forFeature([
      Semester,
      Department,
      Major,
      Class,
      Student,
      StudentProcess,
      StudentCourse,
      Statistic,
      UploadEvent,
      StudentCourse,
      Notification,
      NotificationUser
    ]),
    MulterModule.register({
      dest: './uploads',
    }),
    QueueModule,
    ConfigModule,
  ],
  controllers: [DataController],
  providers: [DataService],
})
export class DataModule {}
