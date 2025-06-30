import { CanActivate, ExecutionContext, Injectable } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { AuthService } from 'src/modules/auth/auth.service';
import { AppJwtService } from 'src/services/app-jwt/app-jwt.service';
import { AuthUtils } from '../utils/auth.utils';
@Injectable()
export class PublicGuard implements CanActivate {
  constructor(
    private readonly authService: AuthService,
    private reflector: Reflector,
    private readonly jwtService: AppJwtService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = AuthUtils.extractTokenFromHeader(request);
    if (!token) return true;

    const user = await this.jwtService.decodeToken(token);

    request['user'] = user;
    return true;
  }
}
