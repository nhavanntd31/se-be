import { CACHE_MANAGER } from '@nestjs/cache-manager';
import { Inject, Injectable, OnModuleInit } from '@nestjs/common';
import { createClient } from 'redis';
import { ConfigService } from 'src/config/config.service';
@Injectable()
export class RedisService implements OnModuleInit {
  constructor(
    private readonly configService: ConfigService,
  ) {}
  private client = createClient({
    socket: {
      host: this.configService.redisConfig.host,
      port: this.configService.redisConfig.port,
    },
  });

  async onModuleInit() {
    await this.client.connect();
  }

  async get(key): Promise<any> {
    try {
      return await this.client.get(key);
    } catch (e) {
      console.error('get', e);
    }
  }

  async set(key, value, time?: number) {
    try {
      await this.client.set(
        key,
        value,
        { EX: time || this.configService.redisConfig.ttl },
      );
    } catch (e) {
      console.error('set', e);
    }
  }
  async reset() {
    await this.client.flushAll();
  }

  async del(key) {
    await this.client.del(key);
  }
  }
