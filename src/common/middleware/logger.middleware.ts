import { Injectable, NestMiddleware, Logger } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import * as util from 'util';

@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  constructor() {}
  private logger = new Logger('HTTP');

  async use(
    request: Request,
    response: Response,
    next: NextFunction,
  ): Promise<void> {
    const start = new Date().getTime();
    const { method, body, originalUrl } = request;
    const ip = request['X-Forwarded-For'] || request['ip'];
    this.logger.log(util.format('PreRequest: %s %s', method, originalUrl));
    response.on('finish', () => {
      this.logger.log(
        util.format(
          '%s %s \n# User: %j \n# Params: %j \n# IP: %s, \n# Process: %s ms',
          method,
          originalUrl,
          request['user'] ? request['user']['id'] : 0,
          body,
          ip,
          new Date().getTime() - start,
        ),
      );
    });
    next();
  }
}
