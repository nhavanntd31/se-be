import { ISendMailOptions } from '@nestjs-modules/mailer';
import { InjectQueue } from '@nestjs/bull';
import { Injectable } from '@nestjs/common';
import { Queue } from 'bull';
import { DEFAUTL_QUEUE_SETTINGS } from 'src/common/constants';

@Injectable()
export class QueueService {
  constructor(
    @InjectQueue('SendMailQueue') private sendMailQueue: Queue,
    @InjectQueue('UpdateDatabaseQueue') private updateDatabaseQueue: Queue,
    @InjectQueue('StatisticQueue') private statisticQueue: Queue,
  ) {}

  async addMailQueue(data: ISendMailOptions) {
    await this.sendMailQueue.add(data, {
      ...DEFAUTL_QUEUE_SETTINGS,
    });
  }

  async addUpdateDatabaseQueue(data: any) {
    await this.updateDatabaseQueue.add(data, {
      ...DEFAUTL_QUEUE_SETTINGS,
    });
  }

  async addStatisticQueue(data: any) {
    await this.statisticQueue.add(data, {
      ...DEFAUTL_QUEUE_SETTINGS,
    });
  }
}
