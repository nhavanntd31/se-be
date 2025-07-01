import * as dotenv from 'dotenv';
import * as Joi from 'joi';

enum DbType {
  mysql = 'mysql',
  postgres = 'postgres',
}

export interface EnvConfig {
  env: string;
}
export interface DBConfigMySQL {
  type: DbType;
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
  log?: boolean;
  synchronize?: boolean;
  migrate?: boolean;
}

export interface S3Config {
  accessKey: string;
  sceret: string;
  bucket: string;
  prefix: string;
}

export interface CloudFrontConfig {
  keyPairId: string;
}

export interface JWTConfig {
  secret: string;
  expiresIn: string;
  refreshExpiresIn: string;
  refreshSecret: string;
}

export interface ProfileExpiredConfig {
  editProfileExpiredTime: number;
}

export interface RedisConfig {
  host: string;
  port: number;
  ttl: number;
  max: number;
}

export interface TwilioConfig {
  sid: string;
  token: string;
  phone: string;
}

export interface NodemailConfig {
  user: string;
  pass: string;
  port: number;
  host: string;
  service: string;
  secure: boolean;
  from: string;
}

export interface GenerateVideoConfig {
  callback: string;
  apiVideo: string;
  token: string;
  apiRecommend: string;
}

export interface TwitterConfig {
  clientId: string;
  secret: string;
}

export interface OnJobConfig {
  status: boolean;
}

export interface CmdKeyConfig {
  key: string;
}

export interface OpenRouterConfig {
  apiKey: string;
  model: string;
  temperature: number;
  systemInstruction: string;
  maxTokens: number;
}

export class ConfigService {
  private readonly envConfig: dotenv.DotenvParseOutput;

  private readonly validationScheme = {
    APP_PORT: Joi.number().default(9000),
    APP_URL: Joi.string().default('/'),
    NODE_ENV: Joi.optional(),

    CORS_ORIGIN: Joi.string().default(''),

    LOG_LEVEL: Joi.string().required(),

    AWS_S3_ACCESS_KEY: Joi.string().default(null),
    AWS_S3_SECRET_KEY: Joi.string().default(null),
    AWS_S3_BUCKET: Joi.string().required(),
    AWS_S3_PREFIX_LINK: Joi.string().required(),

    REDIS_HOST: Joi.string().default(null),
    REDIS_PORT: Joi.string().default(null),
    REDIS_TTL: Joi.string().default(null),

    JWT_SECRET: Joi.string().default('secret'),
    JWT_EXPIRATION_TIME: Joi.string().default('1d'),
    JWT_REFESH_SECRET: Joi.string().default('refresh'),
    JWT_REFESH_EXPIRATION_TIME: Joi.string().default('30d'),

    DB_TYPE: Joi.string().required(),
    DB_HOST: Joi.string().required(),
    DB_PORT: Joi.number().required(),
    DB_USER: Joi.string().required(),
    DB_PASSWORD: Joi.string().required(),
    DB_NAME: Joi.string().required(),
    DB_SYNC: Joi.boolean().default(false),
    DB_MIGRATE: Joi.boolean().default(false),
    DB_LOG: Joi.boolean().default(false),

    DB_SLAVER_HOST: Joi.string().default(null),

    NODE_MAILER_USER: Joi.string().default(null),
    NODE_MAILER_PASS: Joi.string().default(null),
    NODE_MAILER_PORT: Joi.number().default(465),
    NODE_MAILER_HOST: Joi.string().default(null),
    NODE_MAILER_SERVICE: Joi.string().default(null),
    NODE_MAILER_SECURE: Joi.boolean().default(false),
    NODE_MAILER_FROM: Joi.string().default(null),

    ON_JOB: Joi.boolean().default(false),
    CMD_KEY: Joi.string().default(null),

    OPENROUTER_API_KEY: Joi.string().default('sk-or-v1-...'),
    OPENROUTER_MODEL: Joi.string().default('mistralai/mistral-small-3.2-24b-instruct:free'),
    OPENROUTER_TEMPERATURE: Joi.number().default(0.5),
    OPENROUTER_SYSTEM_INSTRUCTION: Joi.string().default('You are a helpful and truthful conversational AI.'),
    OPENROUTER_MAX_TOKENS: Joi.number().default(32000),
  };

  constructor() {
    const configs: dotenv.DotenvParseOutput[] = [];
    const defaultEnvConfigPath = '.env';

    const defaultEnvConfig = dotenv.config({
      path: `${defaultEnvConfigPath}`,
    });
    if (defaultEnvConfig.error) {
      // tslint:disable-next-line: no-console
      console.log(`No config file at path: ${defaultEnvConfigPath}`);
    } else {
      configs.push(defaultEnvConfig.parsed);
      // tslint:disable-next-line: no-console
      console.log(`Loaded config file at path: ${defaultEnvConfigPath}`);
    }
    this.envConfig = this.validateInput(...configs);
  }

  get environmentConfig(): EnvConfig {
    return {
      env: String(this.envConfig.NODE_ENV),
    };
  }

  get jobConfig(): OnJobConfig {
    return {
      status: Boolean(this.envConfig.ON_JOB),
    };
  }

  get cmdKeyConfig(): CmdKeyConfig {
    return {
      key: String(this.envConfig.CMD_KEY),
    };
  }

  get jwtConfig(): JWTConfig {
    return {
      secret: String(this.envConfig.JWT_SECRET),
      expiresIn: String(this.envConfig.JWT_EXPIRATION_TIME),
      refreshSecret: String(this.envConfig.JWT_REFESH_SECRET),
      refreshExpiresIn: String(this.envConfig.JWT_REFESH_EXPIRATION_TIME),
    };
  }

  get corsOrigin(): any {
    return String(this.envConfig.CORS_ORIGIN).trim() === '*'
      ? '*'
      : String(this.envConfig.CORS_ORIGIN).split(',');
  }

  get s3Config(): S3Config {
    return {
      accessKey: String(this.envConfig.AWS_S3_ACCESS_KEY),
      sceret: String(this.envConfig.AWS_S3_SECRET_KEY),
      bucket: String(this.envConfig.AWS_S3_BUCKET),
      prefix: String(this.envConfig.AWS_S3_PREFIX_LINK),
    };
  }

  get redisConfig(): RedisConfig {
    return {
      host: String(this.envConfig.REDIS_HOST),
      port: Number(this.envConfig.REDIS_PORT),
      ttl: Number(this.envConfig.REDIS_TTL),
      max: Number(this.envConfig.REDIS_MAX),
    };
  }

  get cloudfrontConfig(): CloudFrontConfig {
    return {
      keyPairId: String(this.envConfig.CLOUDFRONT_KEY_PAIR_ID),
    };
  }

  get dbConfigMySQL(): DBConfigMySQL {
    return {
      type: this.envConfig.DB_TYPE as DbType,
      host: String(this.envConfig.DB_HOST),
      port: Number(this.envConfig.DB_PORT),
      username: String(this.envConfig.DB_USER),
      password: String(this.envConfig.DB_PASSWORD),
      database: String(this.envConfig.DB_NAME),
      log: Boolean(this.envConfig.DB_LOG),
      synchronize: Boolean(this.envConfig.DB_SYNC),
      migrate: Boolean(this.envConfig.DB_MIGRATE),
    };
  }

  get dbSlaverConfigMySQL(): string[] | null {
    return this.envConfig.DB_SLAVER_HOST
      ? String(this.envConfig.DB_SLAVER_HOST).split(',')
      : null;
  }

  get nodemailConfig(): NodemailConfig {
    return {
      user: String(this.envConfig.NODE_MAILER_USER),
      pass: String(this.envConfig.NODE_MAILER_PASS),
      port: Number(this.envConfig.NODE_MAILER_PORT),
      host: String(this.envConfig.NODE_MAILER_HOST),
      service: String(this.envConfig.NODE_MAILER_SERVICE),
      secure: Boolean(this.envConfig.NODE_MAILER_SECURE),
      from: String(this.envConfig.NODE_MAILER_FROM),
    };
  }

  get port(): number {
    return Number(this.envConfig.APP_PORT);
  }
  get basePath(): string {
    return this.envConfig.APP_URL;
  }

  get openRouterConfig(): OpenRouterConfig {
    return {
      apiKey: String(this.envConfig.OPENROUTER_API_KEY),
      model: String(this.envConfig.OPENROUTER_MODEL),
      temperature: Number(this.envConfig.OPENROUTER_TEMPERATURE),
      systemInstruction: String(this.envConfig.OPENROUTER_SYSTEM_INSTRUCTION),
      maxTokens: Number(this.envConfig.OPENROUTER_MAX_TOKENS),
    };
  }

  public get(key: string): string {
    return process.env[key];
  }

  public getNumber(key: string): number {
    return Number(this.get(key));
  }

  private validateInput(
    ...envConfig: dotenv.DotenvParseOutput[]
  ): dotenv.DotenvParseOutput {
    const mergedConfig: dotenv.DotenvParseOutput = {};

    envConfig.forEach((config) => Object.assign(mergedConfig, config));

    const envVarsSchema: Joi.ObjectSchema = Joi.object(this.validationScheme);

    const result = envVarsSchema.validate(mergedConfig);
    if (result.error) {
      throw new Error(`Config validation error: ${result.error.message}`);
    }
    return result.value;
  }
}
