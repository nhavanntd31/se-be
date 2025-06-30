import { Controller } from '@nestjs/common';
import { AppJwtService } from './app-jwt.service';

@Controller('app-jwt')
export class AppJwtController {
  constructor(private readonly appJwtService: AppJwtService) {}
}
