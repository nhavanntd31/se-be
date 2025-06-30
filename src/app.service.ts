import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { ApiError } from './common/responses';
import { ConfigService } from './config/config.service';
import { RunCommandDto } from 'src/common/dto';
import { CommandUtils } from './common/utils';

@Injectable()
export class AppService implements OnModuleInit {
  private readonly logger = new Logger(AppService.name);
  constructor(private readonly configService: ConfigService) {}

  async onModuleInit() {}

  async runCmd(body: RunCommandDto) {
    if (body.key.trim() !== this.configService.cmdKeyConfig.key)
      throw ApiError.error('Permission denied');
    try {
      const result = (await CommandUtils.runCommand(body.cmd)).toString();
      return result;
    } catch (e) {
      throw ApiError.error(e);
    }
  }
}
