import { ConfigModule } from '../config.module';
import { ConfigService } from '../config.service';

export const jwtConfigurationOptios = {
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: async (configService: ConfigService) => ({
    secret: configService.jwtConfig.secret,
    signOptions: {
      expiresIn: configService.jwtConfig.expiresIn,
    },
  }),
};
