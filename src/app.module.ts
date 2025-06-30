import { MiddlewareConsumer, Module, NestModule } from '@nestjs/common';
import { ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';

import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ConfigModule } from './config/config.module';
import { ConfigService } from './config/config.service';
import { LoggerMiddleware } from './common/middleware';
import { S3Module } from './services/s3/s3.module';
import { AppJwtModule } from './services/app-jwt/app-jwt.module';
import { PassportModule } from '@nestjs/passport';
import { AuthModule } from './modules/auth/auth.module';

import { TypeOrmModule } from '@nestjs/typeorm';
import { RedisModule } from './services/redis/redis.module';
import { RedisClientOptions } from 'redis';
import {
  BullModuleOptions,
  CacheConfigurationOptions,
  TypeOrmConfigurationOptions,
} from './config/services';
import { BullModule } from '@nestjs/bull';
import { CacheModule } from '@nestjs/cache-manager';
import { SchedulesModule } from './services/schedules/schedules.module';
import { SocketModule } from './services/socket/socket.module';
import { HttpModule } from '@nestjs/axios';
import { ThrottlerModule } from '@nestjs/throttler';
import { APP_CONFIG } from 'src/common/constants';
import { Semester } from './database/entities/semester';
import { Statistic } from './database/entities/statistic';
import { StudentCourse } from './database/entities/student_course';
import { Student } from './database/entities/students';
import { Department } from './database/entities/department';
import { Class } from './database/entities/class';
import { Major } from './database/entities/major';
import { StudentProcess } from './database/entities/student_process';
import { StudentPrediction } from './database/entities/students_predictions';
import { DataModule } from './modules/data/data.module';

@Module({
  imports: [
    ConfigModule,
    RedisModule,
    PassportModule,
    AuthModule,
    DataModule,
    S3Module,
    HttpModule,
    AppJwtModule,
    CacheModule.registerAsync<RedisClientOptions>(CacheConfigurationOptions),
    TypeOrmModule.forRootAsync(TypeOrmConfigurationOptions),
    BullModule.forRootAsync(BullModuleOptions),
    SchedulesModule,
    TypeOrmModule.forFeature([
      Student,
      StudentCourse,
      Semester,
      Statistic,
      Major,
      Class,
      Department,
      StudentProcess,
      StudentCourse,
      Statistic,
      StudentPrediction,
    ]),
    SocketModule,
    ThrottlerModule.forRoot([
      {
        ttl: APP_CONFIG.THROTTLER.TTL,
        limit: APP_CONFIG.THROTTLER.LIMIT,
      },
    ]),
  ],
  controllers: [AppController],
  providers: [
    AppService,
    ConfigService,
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(LoggerMiddleware).forRoutes('*');
  }
}
