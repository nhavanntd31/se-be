import { TypeOrmModuleAsyncOptions } from '@nestjs/typeorm';
import { join } from 'path';
import { ConfigModule } from '../config.module';
import { ConfigService } from '../config.service';

export const TypeOrmConfigurationOptions: TypeOrmModuleAsyncOptions = {
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: async (configService: ConfigService) => ({
    type: configService.dbConfigMySQL.type,
    host: configService.dbConfigMySQL.host,
    port: configService.dbConfigMySQL.port,
    database: configService.dbConfigMySQL.database,
    username: configService.dbConfigMySQL.username,
    password: configService.dbConfigMySQL.password,
    synchronize: configService.dbConfigMySQL.synchronize,
    extra: {
      charset: 'utf8mb4_general_ci',
    },
    entities: [join(__dirname, 'database', 'entities', '*.entity.{ts,js}')],
    migrationsRun: configService.dbConfigMySQL.migrate,
    migrationsTransactionMode: 'each',
    migrations: [join(__dirname, 'database', 'migrations', '*.{ts,js}')],
    logging: configService.dbConfigMySQL.log,
    timezone: 'Z',
    autoLoadEntities: true,
    cli: {
      migrationsDir: join(__dirname, 'database', 'migrations'),
    },
  }),
};
