import { Seeder, SeederFactoryManager, runSeeders } from 'typeorm-extension';
import { DataSource } from 'typeorm';
import { UserFactory } from '../factories';
import UserSeeder from './user.seed';

export default class InitSeeder implements Seeder {
  public async run(
    dataSource: DataSource,
    factoryManager: SeederFactoryManager,
  ): Promise<any> {
    await runSeeders(dataSource, {
      seeds: [UserSeeder],
      factories: [UserFactory],
    });
  }
}
