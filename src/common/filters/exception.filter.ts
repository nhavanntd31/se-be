import {
  ArgumentsHost,
  BadRequestException,
  Catch,
  ExceptionFilter,
  HttpException,
  Logger,
  UnauthorizedException,
} from '@nestjs/common';
import { ConfigService } from 'src/config/config.service';
import * as util from 'util';

import { ApiError } from '../responses/api-error';
import { ErrorDto } from '../responses/error-exception';
import { LoggerUtils } from '../utils';
import { ERROR_CODE } from 'src/common/constants';

@Catch()
export class ApiErrorFilter implements ExceptionFilter {
  private logger = new Logger('HTTP_ERROR');

  constructor(private readonly configService: ConfigService) {}

  async catch(
    exception:
      | HttpException
      | ApiError
      | UnauthorizedException
      | BadRequestException,
    host: ArgumentsHost,
  ) {
    const ctx = host.switchToHttp();
    const request = ctx.getRequest<Request>();
    const response = ctx.getResponse();
    let statusCode = 500;
    const errorDto = new ErrorDto();

    if (exception instanceof ApiError) {
      statusCode = exception.getStatus();
      errorDto.meta = exception.meta;
    } else if (exception instanceof UnauthorizedException) {
      statusCode = exception.getStatus();
      errorDto.message = exception.message;
    } else if (exception instanceof BadRequestException) {
      statusCode = exception.getStatus();
      errorDto.meta = {
        msg: exception.getResponse()['message'],
        code: -1,
        errorCode: exception.getResponse()['errorCode'] || ERROR_CODE.DEFAULT,
        extraInfo: {
          ...exception.getResponse()['extraInfo'],
        },
      };
    } else if (exception instanceof HttpException) {
      statusCode = exception.getStatus();
      errorDto.meta = {
        msg: exception.getResponse()['message'],
        code: -1,
        errorCode: 'E0',
        extraInfo: exception.getResponse(),
      };
    } else {
      errorDto.meta = {
        message: util.inspect(exception),
        code: -1,
        errorCode: 'E0',
      };
    }
    // if (
    //   this.configService.environmentConfig.env !== 'prod' &&
    //   this.configService.environmentConfig.env !== 'production' &&
    //   this.configService.environmentConfig.env !== 'stg' &&
    //   this.configService.environmentConfig.env !== 'staging'
    // ) {
    //   LoggerUtils.getLogger(request).error(
    //     util.inspect(errorDto.meta?.msg || errorDto),
    //   );
    // }
    this.logger.error(util.inspect(errorDto.meta));
    response.status(statusCode).json(errorDto);
  }
}
