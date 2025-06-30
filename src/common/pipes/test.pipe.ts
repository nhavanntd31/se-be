import { PipeTransform, Injectable, BadRequestException } from '@nestjs/common';

@Injectable()
export class ParseObjectIdPipe implements PipeTransform<any, any> {
  transform(value: any, data): any {}
}
