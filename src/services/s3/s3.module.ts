import { Module } from '@nestjs/common';
import { S3Service } from './s3.service';
import { ConfigModule } from 'src/config/config.module';

@Module({
  providers: [S3Service],
  imports: [ConfigModule],
  exports: [S3Service],
})
export class S3Module {}
