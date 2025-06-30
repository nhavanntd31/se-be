// import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';
import { MailConfig } from 'src/common/constants';
import { ConfigService } from 'src/config/config.service';
import { QueueService } from '../queue/queue.service';

@Injectable()
export class MailService {
  constructor(
    // private readonly mailerService: MailerService,
    private readonly configService: ConfigService,
    private readonly queueService: QueueService,
  ) {}

  async sendVerifiedRegisterCode(email: string, code: string) {
    await this.queueService.addMailQueue({
      to: email,
      subject: MailConfig.EMAIL_SUBJECT.REGISTER,
      template: './register', // `.hbs` extension is appended automatically
      context: {
        code,
      },
    });
  }

  async resendVerifiedRegisterCode(email: string, code: string) {
    await this.queueService.addMailQueue({
      to: email,
      subject: MailConfig.EMAIL_SUBJECT.RESEND_REGISTER,
      template: './register', // `.hbs` extension is appended automatically
      context: {
        code,
      },
    });
  }

  async sendForgotPasswordConfirmation(email: string, code: string) {
    await this.queueService.addMailQueue({
      to: email,
      subject: MailConfig.EMAIL_SUBJECT.RESET_PASSWORD,
      template: './forgot-password', // `.hbs` extension is appended automatically
      context: {
        code,
      },
    });
  }
}
