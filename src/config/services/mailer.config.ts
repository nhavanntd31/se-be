import { join } from 'path';
import { ConfigModule } from '../config.module';
import { ConfigService } from '../config.service';
import { HandlebarsAdapter } from '@nestjs-modules/mailer/dist/adapters/handlebars.adapter';
import { MailerAsyncOptions } from '@nestjs-modules/mailer/dist/interfaces/mailer-async-options.interface';

export const MailerConfigurationOptions: MailerAsyncOptions = {
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: async (configService: ConfigService) => ({
    transport: {
      service: configService.nodemailConfig.service,
      port: configService.nodemailConfig.port,
      host: configService.nodemailConfig.host,
      auth: {
        user: configService.nodemailConfig.user,
        pass: configService.nodemailConfig.pass,
      },
    },
    defaults: {
      from: `<${configService.nodemailConfig.from}>`,
    },
    template: {
      dir: join(__dirname, '..', '..', 'services', 'mail', 'templates'),
      adapter: new HandlebarsAdapter(), // or new PugAdapter() or new EjsAdapter()
      options: {
        strict: true,
      },
    },
  }),
};
