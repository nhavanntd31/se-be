import { DataSource, DataSourceOptions } from 'typeorm';
import { ConfigService } from '../config.service';

const configService = new ConfigService();
const dataSourceOptions: DataSourceOptions = {
  type: 'postgres',

  host: configService.dbConfigMySQL.host,
  port: Number(configService.dbConfigMySQL.port),
  
  database: configService.dbConfigMySQL.database,
  username: configService.dbConfigMySQL.username,
  password: configService.dbConfigMySQL.password,
  synchronize: configService.dbConfigMySQL.synchronize,
  extra: {
    charset: 'utf8mb4_general_ci',
  },
  entities: ['dist/database/entities/*{.js,.ts}'],
  migrationsTransactionMode: 'each',
  migrations: ['dist/database/migrations/*.js'],
  logging: true,
};

const typeOrmMigration = new DataSource(dataSourceOptions);

export default typeOrmMigration;
