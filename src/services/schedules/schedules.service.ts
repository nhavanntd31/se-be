import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { InjectRepository } from '@nestjs/typeorm';
import { ConfigService } from 'src/config/config.service';
import { User } from 'src/database/entities';
import { Repository } from 'typeorm';
import { RedisService } from '../redis/redis.service';

@Injectable()
export class SchedulesService implements OnModuleInit {
  private readonly logger = new Logger('SchedulesService');
  constructor(
    private readonly redisService: RedisService,
    private readonly configService: ConfigService,
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
  ) {}

  async onModuleInit() {}

  @Cron(CronExpression.EVERY_10_MINUTES)
  async test() {
    if (!this.configService.jobConfig.status) return;
    this.logger.debug('======Start job test()======');
    this.logger.debug('======End job test()======');
  }
}
