import { ConfigModule } from '../config.module';
import { ConfigService } from '../config.service';

export const BullModuleOptions = {
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: async (configService: ConfigService) => ({
    redis: {
      host: configService.redisConfig.host,
      port: configService.redisConfig.port,
    },
  }),
};