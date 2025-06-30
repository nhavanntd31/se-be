import { NestFactory } from '@nestjs/core';
import { LogLevel } from '@nestjs/common';
import 'dotenv/config';

import { AppModule } from './app.module';
import { ApiErrorFilter } from './common/filters/exception.filter';
import { setupSwagger } from './common/swagger';
import { ConfigService } from './config/config.service';
import { RedisIoAdapter } from './services/redis/adapters/redis.adaptor';
import { customValidationPipe } from 'src/common/validation-pipe';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    cors: true,
    logger: process.env.LOG_LEVEL.split(',') as LogLevel[],
  });
  const configService = app.get(ConfigService);

  app.enableCors({
    origin: configService.corsOrigin,
  });

  const redisIoAdapter = new RedisIoAdapter(app, configService);
  await redisIoAdapter.connectToRedis();

  app.useWebSocketAdapter(redisIoAdapter);
  app.useGlobalFilters(new ApiErrorFilter(configService));
  app.setGlobalPrefix('api');

  setupSwagger(app);
  customValidationPipe(app);
  await app.listen(configService.port);
  console.log(`Listen at port: ${configService.port}`);
}
bootstrap();
