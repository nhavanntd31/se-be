import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Post,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation } from '@nestjs/swagger';
import { AppService } from './app.service';
import { Roles } from './common/decorators/role.decorator';
import { ApiOK } from './common/responses';
import { UserRole } from './database/entities';
import { RunCommandDto } from 'src/common/dto';
import { JwtAuthGuard, RolesGuard } from 'src/common/guards';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get('')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Health check' })
  async health() {
    return ApiOK.success({ result: true });
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.ADMIN)
  @Post('cmd')
  @HttpCode(HttpStatus.OK)
  @ApiBearerAuth()
  @ApiOperation({ description: 'Command line' })
  async runCmd(@Body() body: RunCommandDto) {
    const result = await this.appService.runCmd(body);
    return new ApiOK(result);
  }
}
