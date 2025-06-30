import { Global, Module } from '@nestjs/common';
import { AppJwtService } from './app-jwt.service';
import { AppJwtController } from './app-jwt.controller';
import { jwtConfigurationOptios } from 'src/config/services';
import { JwtModule } from '@nestjs/jwt';
import { ConfigModule } from 'src/config/config.module';

@Global()
@Module({
  controllers: [AppJwtController],
  providers: [AppJwtService],
  imports: [JwtModule.registerAsync(jwtConfigurationOptios), ConfigModule],
  exports: [AppJwtService],
})
export class AppJwtModule {}
