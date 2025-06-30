import { INestApplication } from '@nestjs/common';
import {
  DocumentBuilder,
  SwaggerDocumentOptions,
  SwaggerModule,
} from '@nestjs/swagger';
import { ConfigService } from 'src/config/config.service';

export function setupSwagger(app: INestApplication) {
  const configService = app.get(ConfigService);
  app.setGlobalPrefix('api');
  const config = new DocumentBuilder()
    .setTitle('Eden Video')
    .setDescription('The app API description')
    .setVersion('1.0')
    .addServer(configService.basePath)
    .addBearerAuth()
    .build();

  const options: SwaggerDocumentOptions = {
    operationIdFactory: (controllerKey: string, methodKey: string) => methodKey,
  };

  const document = SwaggerModule.createDocument(app, config, options);
  SwaggerModule.setup('docs', app, document);
}
