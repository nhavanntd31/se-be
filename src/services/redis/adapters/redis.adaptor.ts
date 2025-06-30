import { IoAdapter } from '@nestjs/platform-socket.io';
import { ServerOptions } from 'socket.io';
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';
import { ConfigService } from 'src/config/config.service';

export class RedisIoAdapter extends IoAdapter {
  private adapterConstructor: ReturnType<typeof createAdapter>;
  constructor(app, private readonly configService: ConfigService) {
    super(app);
  }

  async connectToRedis(): Promise<void> {
    try {
      const pubClient = createClient({
        socket: {
          host: this.configService.redisConfig.host,
          port: this.configService.redisConfig.port,
        },
      });
      const subClient = pubClient.duplicate();

      await Promise.all([pubClient.connect(), subClient.connect()]);

      this.adapterConstructor = createAdapter(pubClient, subClient);
    } catch (e) {
      console.error(e);
    }
  }

  createIOServer(port: number, options?: ServerOptions): any {
    const server = super.createIOServer(port, options);
    server.adapter(this.adapterConstructor);
    return server;
  }
}
