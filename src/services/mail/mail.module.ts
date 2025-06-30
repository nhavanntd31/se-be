import { MailerModule } from '@nestjs-modules/mailer';
import { Module } from '@nestjs/common';
import { MailService } from './mail.service';
import { ConfigModule } from 'src/config/config.module';
import { MailerConfigurationOptions } from 'src/config/services/mailer.config';
import { QueueModule } from '../queue/queue.module';

@Module({
  imports: [
    MailerModule.forRootAsync(MailerConfigurationOptions),
    ConfigModule,
    QueueModule,
  ],
  providers: [MailService],
  exports: [MailService],
})
export class MailModule {}
