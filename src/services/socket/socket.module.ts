import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule } from 'src/config/config.module';
import { User } from 'src/database/entities';
import { SocketGateway } from './socket.gateway';

@Module({
  imports: [TypeOrmModule.forFeature([User]), JwtModule, ConfigModule],
  providers: [SocketGateway],
  exports: [SocketGateway],
})
export class SocketModule {}
