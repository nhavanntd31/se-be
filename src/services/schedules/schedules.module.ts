import { HttpModule } from '@nestjs/axios';
import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule } from 'src/config/config.module';
import { User } from 'src/database/entities';
import { RedisModule } from '../redis/redis.module';
import { SocketModule } from '../socket/socket.module';
import { SchedulesController } from './schedules.controller';
import { SchedulesService } from './schedules.service';

@Module({
  controllers: [SchedulesController],
  providers: [SchedulesService],
  imports: [
    RedisModule,
    HttpModule,
    ConfigModule,
    SocketModule,
    ScheduleModule.forRoot(),
    TypeOrmModule.forFeature([User]),
  ],
  exports: [SchedulesService],
})
export class SchedulesModule {}
