import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
import { ConfigService } from 'src/config/config.service';
import { User, UserTokenPayload } from 'src/database/entities';
import { APP_CONFIG } from '../constants/app-config';

export class AuthUtils {
  public static hashData(data: string) {
    return bcrypt.hash(data, APP_CONFIG.SALT_ROUND);
  }

  public static async compareData(hash: string, data: string) {
    return await bcrypt.compare(data, hash);
  }

  public static async generateToken(
    payload: UserTokenPayload,
    jwtService: JwtService,
    configService: ConfigService,
  ) {
    const accessToken = await jwtService.signAsync(payload, {
      secret: configService.jwtConfig.secret,
      expiresIn: configService.jwtConfig.expiresIn,
    });
    const refreshToken = await jwtService.signAsync(payload, {
      secret: configService.jwtConfig.refreshSecret,
      expiresIn: configService.jwtConfig.refreshExpiresIn,
    });

    return { accessToken, refreshToken };
  }

  public static extractTokenFromHeader(
    request: Request | Record<string, any>,
  ): string | undefined {
    const [type, token] = request.headers['X-API-JWT-TOKEN']?.split(' ') ?? [];
    return type === 'Bearer' && token != 'null' && token != 'undefined'
      ? token
      : undefined;
  }

  public static generateCode(num: number) {
    const n = 10 ** num;
    return Math.floor(Math.random() * n)
      .toString()
      .padStart(num, '0');
  }
}
