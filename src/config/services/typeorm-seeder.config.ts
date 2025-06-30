import { DataSource, DataSourceOptions } from 'typeorm';
import { ConfigService } from '../config.service';
import { SeederOptions } from 'typeorm-extension';
import InitSeeder from 'src/database/seeding/seeds/main.seed';

const configService = new ConfigService();
const dataSourceOptions = {
  type: configService.dbConfigMySQL.type,
  host: configService.dbConfigMySQL.host,
  port: Number(configService.dbConfigMySQL.port),
  database: configService.dbConfigMySQL.database,
  username: configService.dbConfigMySQL.username,
  password: configService.dbConfigMySQL.password,
  entities: ['dist/database/entities/*{.js,.ts}'],
  logging: true,
  migrationsTransactionMode: 'each',
  migrations: ['dist/database/migrations/*.js'],
  seeds: [InitSeeder],
};

export const dataSource = new DataSource(
  dataSourceOptions as DataSourceOptions & SeederOptions,
);
