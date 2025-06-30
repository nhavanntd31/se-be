import { Global, Module } from '@nestjs/common';
import { RedisService } from './redis.service';
import { ConfigModule } from 'src/config/config.module';
import { RedisController } from './redis.controller';
import { RedisIoAdapter } from './adapters/redis.adaptor';

@Global()
@Module({
  providers: [RedisService, RedisIoAdapter],
  imports: [ConfigModule],
  exports: [RedisService],
  controllers: [RedisController],
})
export class RedisModule {}
