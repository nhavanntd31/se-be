import { Global, Module } from '@nestjs/common';
import { AuthService } from './auth.service';
import { AuthController } from './auth.controller';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule } from 'src/config/config.module';
import { JwtModule } from '@nestjs/jwt';
import { JwtStrategy } from 'src/common/strategies';
import { RedisModule } from 'src/services/redis/redis.module';
import { User } from 'src/database/entities';
import { HttpModule } from '@nestjs/axios';
import { MailModule } from 'src/services/mail/mail.module';

@Global()
@Module({
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy],
  imports: [
    TypeOrmModule.forFeature([User]),
    ConfigModule,
    JwtModule,
    RedisModule,
    HttpModule,
    MailModule,
  ],
  exports: [AuthService],
})
export class AuthModule {}
