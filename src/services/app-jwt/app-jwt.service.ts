import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from 'src/config/config.service';

@Injectable()
export class AppJwtService {
  constructor(
    private readonly jwtService: JwtService,
    private readonly configService: ConfigService,
  ) {}

  async verifyToken(token: string) {
    try {
      const payload = await this.jwtService.verifyAsync(token, {
        secret: this.configService.jwtConfig.secret,
      });
      return payload;
    } catch {
      throw new UnauthorizedException();
    }
  }

  async decodeToken(token: string) {
    try {
      const payload = this.jwtService.decode(token);
      return payload;
    } catch {
      throw new UnauthorizedException();
    }
  }
}
