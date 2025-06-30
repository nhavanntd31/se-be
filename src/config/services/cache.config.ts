import { redisStore } from 'cache-manager-redis-yet';
import { ConfigModule } from '../config.module';
import { ConfigService } from '../config.service';

export const CacheConfigurationOptions = {
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: async (configService: ConfigService) => {
    return {
      store: redisStore,
      host: configService.redisConfig.host,
      port: configService.redisConfig.port,
      ttl: configService.redisConfig.ttl,
    };
  },
  isGlobal: true,
};
