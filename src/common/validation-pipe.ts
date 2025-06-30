import {
  BadRequestException,
  INestApplication,
  ValidationError,
  ValidationPipe,
} from '@nestjs/common';
import * as _ from 'lodash';

export function customValidationPipe(app: INestApplication) {
  app.useGlobalPipes(
    new ValidationPipe({
      transform: true,
      forbidUnknownValues: false,
      whitelist: true,
      exceptionFactory: (errors: ValidationError[]) => {
        const extraInfo = _.reduce(
          errors,
          (data, item) => {
            const values = Object.values(item.constraints);
            const tmp = {
              field: item.property,
              value: values[values.length - 1],
            };
            data.push(tmp);
            return data;
          },
          [],
        );
        const code = extraInfo[0].value;
        delete extraInfo[0].value;
        const response = {
          errorCode: code,
          extraInfo: extraInfo[0],
          message: 'Invalid params',
        };
        return new BadRequestException(response);
      },
    }),
  );
}
