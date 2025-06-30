import { Module } from '@nestjs/common';
import { TasksService } from './tasks.service';
import { ConfigService } from 'src/config/config.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { BullModule } from '@nestjs/bull';
import { ConfigModule } from 'src/config/config.module';
import { QueueService } from './queue.service';
import { S3Module } from '../s3/s3.module';
import { SocketModule } from '../socket/socket.module';
import { RedisModule } from '../redis/redis.module';
import { HttpModule } from '@nestjs/axios';
import { SendMailQueueProcessor, UpdateDatabaseQueueProcessor } from './processors';
import { StudentProcess } from 'src/database/entities/student_process';
import { StudentCourse } from 'src/database/entities/student_course';
import { Semester } from 'src/database/entities/semester';
import { Student } from 'src/database/entities/students';
import { Department } from 'src/database/entities/department';
import { Class } from 'src/database/entities/class';
import { Major } from 'src/database/entities/major';
import { Statistic } from 'src/database/entities/statistic';
import { StatisticQueueProcessor } from './processors/statistic.processor';
import { UploadEvent } from 'src/database/entities/upload_event';

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
    ]),
    ConfigModule,
    S3Module,
    SocketModule,
    RedisModule,
    HttpModule,
    BullModule.registerQueueAsync({
      name: 'SendMailQueue',
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: async (configService: ConfigService) => ({
        redis: {
          host: configService.redisConfig.host,
          port: configService.redisConfig.port,
        },
      }),
    }),
    BullModule.registerQueueAsync({
      name: 'UploadS3Queue',
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: async (configService: ConfigService) => ({
        redis: {
          host: configService.redisConfig.host,
          port: configService.redisConfig.port,
        },
      }),
    }),
    BullModule.registerQueueAsync({
      name: 'UpdateDatabaseQueue',
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: async (configService: ConfigService) => ({
        redis: {
          host: configService.redisConfig.host,
          port: configService.redisConfig.port,
        },
      }),
    }),
    BullModule.registerQueueAsync({
      name: 'StatisticQueue',
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: async (configService: ConfigService) => ({
        redis: {
          host: configService.redisConfig.host,
          port: configService.redisConfig.port,
        },
      }),
    }),
  ],
  providers: [TasksService, QueueService, SendMailQueueProcessor, UpdateDatabaseQueueProcessor, StatisticQueueProcessor],
  exports: [QueueService],
})
export class QueueModule {}
