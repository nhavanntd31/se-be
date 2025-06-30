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

@Processor('StatisticQueue')
export class StatisticQueueProcessor {
  private readonly logger = new Logger('StatisticQueue');

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
    this.logger.error(
      `Processor:@OnQueueFailed - Failed job ${job.id} of type ${job.queue.name}: ${error.message}`,
      error.stack,
    );
  }

  @Process()
  async handle(job: Job): Promise<any> {
    this.logger.log('Processor:@Process - Statistic.');
    try {
      this.logger.log('Statistic process started');
      await this.taskService.statisticProcess(job);
    } catch (error) {
      this.logger.error('Failed to statistic.', error.stack);
      throw error;
    }
  }
}