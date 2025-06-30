import { CanActivate, ExecutionContext, Injectable } from '@nestjs/common';
import { ApiKeyForExtention } from '../constants';
import { ApiError } from '../responses';
import { AuthUtils } from '../utils';
@Injectable()
export class ExtendApiGuard implements CanActivate {
  constructor() {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = AuthUtils.extractTokenFromHeader(request);

    if (!token) throw ApiError.error('Unauthorized', '401');

    if (ApiKeyForExtention.indexOf(token) == -1)
      throw ApiError.error('Forbidden', '403');

    return true;
  }
}
