import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import { IsNumber, IsOptional } from 'class-validator';

export const APP_CONFIG = {
  SALT_ROUND: 10,

  STUDENT_GRADUATE_CREDIT: 125,
  // Video
  RATE_LIMIT: {
    limit: 1,
    ttl: 1,
  },
  THROTTLER: {
    LIMIT: 20,
    TTL: 10000,
  },
  OTP_SETTING: {
    EXPIRED: 30 * 60, // time to senconds
  },

  // Login 3rd
  GOOGLE_USER_PROFILE_URL: 'https://www.googleapis.com/oauth2/v2/userinfo',
};

export const DEFAUTL_QUEUE_SETTINGS = {
  attempts: 1,
  removeOnComplete: true,
  backoff: 5000,
  timeout: 0
};

export const REDIS_KEY = {
  USER_INFO: 'user',
};

export const ApiKeyForExtention = [
  '10DZaFA7S7C5dly2eWzOlBnuZ9e7m43EoAyFIr9stZk8IgwHxnEdzy',
];

export class BaseSearchDto {
  @ApiProperty({ required: false })
  @IsOptional()
  keyword?: string;

  @ApiProperty({ required: false })
  @IsOptional()
  sortField?: string;

  @ApiProperty({ required: false, description: '-1: DESC, 1: ASC' })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  sortType?: number;

  @ApiProperty({ required: false })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  offset?: number;

  @ApiProperty({ required: false })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  limit?: number;
}
