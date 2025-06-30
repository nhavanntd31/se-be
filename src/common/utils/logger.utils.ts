/* eslint-disable @typescript-eslint/no-var-requires */
const { createLogger, format, transports } = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
import { ConfigService } from 'src/config/config.service';
import * as util from 'util';

const { combine, timestamp, printf } = format;

export class LoggerUtils {
  constructor(private readonly configService: ConfigService) {}

  public static getLogger(request?: any) {
    const formatLog = printf(({ level, message, timestamp }) => {
      return util.format(
        `%s %s %s  \n# Params: %j \n# ${level}: %s`,
        timestamp,
        request?.['method'],
        request?.['url'],
        request?.['body'],
        message,
      );
    });

    const transportApi = new DailyRotateFile({
      filename: '.logs/%DATE%.error.log',
      datePattern: 'YYYY-MM-DD',
      level: process.env.LOG_LEVEL,
      maxSize: '15m',
    });

    const transportConsole = new transports.Console({
      colorize: true,
      timestamp: true,
    });

    const logger = createLogger({
      format: combine(timestamp(), formatLog),
      transports: [transportConsole, transportApi],
    });

    return logger;
  }
}
