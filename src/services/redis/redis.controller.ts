import {
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpStatus,
  Query,
} from '@nestjs/common';
import { ApiExcludeEndpoint, ApiProperty, ApiTags } from '@nestjs/swagger';
import { RedisService } from './redis.service';

@Controller('redis')
@ApiTags('Redis')
export class RedisController {
  constructor(private readonly redisService: RedisService) {}

  @ApiExcludeEndpoint()
  @Get()
  @HttpCode(HttpStatus.OK)
  @ApiProperty({})
  async getRedisValue(@Query('key') key: string) {
    const result = await this.redisService.get(key);
    return result;
  }
}
