import { Controller, Get, HttpCode, HttpStatus } from '@nestjs/common';
import { ApiProperty, ApiTags } from '@nestjs/swagger';
import { SchedulesService } from './schedules.service';

@ApiTags('schedule')
@Controller('schedules')
export class SchedulesController {
  constructor(private readonly schedulesService: SchedulesService) {}
}
