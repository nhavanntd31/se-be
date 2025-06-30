import {
  OnQueueActive,
  OnQueueCompleted,
  OnQueueFailed,
  Process,
  Processor,
} from '@nestjs/bull';
import { Logger } from '@nestjs/common';
import { Job } from 'bull';
import { TasksService } from '../tasks.service';

@Processor('SendMailQueue')
export class SendMailQueueProcessor {
  private readonly logger = new Logger('SendMailQueue');

  constructor(private readonly taskService: TasksService) {}

  @OnQueueActive()
  onActive(job: Job) {
    this.logger.log(
      `Processor:@OnQueueActive - Processing job ${job.id} of type ${job.queue.name}.`,
    );
  }

  @OnQueueCompleted()
  onComplete(job: Job) {
    this.logger.log(
      `Processor:@OnQueueCompleted - Completed job ${job.id} of type ${job.queue.name}.`,
    );
  }

  @OnQueueFailed()
  onError(job: Job<any>, error) {
    console.log(error);
    this.logger.log(
      `Processor:@OnQueueFailed - Failed job ${job.id} of type ${job.queue.name}: ${error.message}`,
      error.stack,
    );
  }

  @Process()
  async handle(job: Job): Promise<any> {
    this.logger.log('Processor:@Process - Send email.');
    try {
      // processing here
      await this.taskService.sendMailProcess(job);
    } catch (error) {
      this.logger.error('Failed to send email.', error.stack);
      throw error;
    }
  }
}
