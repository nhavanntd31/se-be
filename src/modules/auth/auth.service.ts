import { HttpService } from '@nestjs/axios';
import { Injectable, Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import { AxiosResponse } from 'axios';

import { APP_CONFIG, REDIS_KEY, ERROR_CODE } from 'src/common/constants';
import { ApiError, ApiOK } from 'src/common/responses';
import { AuthUtils, Utils } from 'src/common/utils';
import { ConfigService } from 'src/config/config.service';
import {
  User,
  UserAuthenType,
  UserStatus,
  UserTokenPayload,
} from 'src/database/entities';
import { MailService } from 'src/services/mail/mail.service';
import { RedisService } from 'src/services/redis/redis.service';
import { Repository } from 'typeorm';
import {
  ForgotPasswordDto,
  LoginAppDto,
  LoginEmailDto,
  RegisterEmailDto,
  ResetPasswordDto,
  ResendCodeDto,
  VerifyEmailRegisterDto,
  ChangePasswordDto,
} from './dto';

@Injectable()
export class AuthService {
  private logger = new Logger('AuthService');
  constructor(
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
    private readonly jwtService: JwtService,
    private readonly configService: ConfigService,
    private readonly redisService: RedisService,
    private readonly httpService: HttpService,
    private readonly mailService: MailService,
  ) {}

  async registerByEmail(body: RegisterEmailDto) {
    // check email exist
    const isExists = await this.userRepository.findOne({
      where: {
        email: body.email.trim(),
        isDeleted: false,
      },
    });
    if (isExists)
      throw ApiError.error('Email exist', ERROR_CODE.EMAIL_EXIST, {
        fiell: 'email',
      });

    // generate code for verify
    const code = AuthUtils.generateCode(6);

    // Create user
    await this.userRepository.save({
      email: body.email,
      name: body.name,
      password: await AuthUtils.hashData(body.password),
      otpCode: code,
      otpCodeCreatedAt: new Date(),
      isVerified: false,
    });

    // send email
    this.mailService.sendVerifiedRegisterCode(body.email, code);
    return ApiOK.success({ result: true });
  }

  async verifyRegister(body: VerifyEmailRegisterDto) {
    // check email exist
    const user = await this.userRepository.findOne({
      where: {
        email: body.email.trim(),
        isVerified: false,
        isDeleted: false,
      },
    });
    if (!user)
      throw ApiError.error('Email not exist', ERROR_CODE.EMAIL_NOT_REGISTER, {
        fiell: 'email',
      });

    if (user.otpCode !== body.code)
      throw ApiError.error('Code invalid.', ERROR_CODE.CODE_INVALID);

    if (
      Utils.addDateTime(user.otpCodeCreatedAt, APP_CONFIG.OTP_SETTING.EXPIRED) <
      Utils.utcUnixDateTime()
    )
      throw ApiError.error('Code expired', ERROR_CODE.CODE_INVALID);

    user.otpCode = null;
    user.otpCodeCreatedAt = null;
    user.isVerified = true;
    user.status = UserStatus.ACTIVE;

    const payload = {
      id: user.id,
      email: user.email,
      role: user.role,
    } as UserTokenPayload;
    const token = await AuthUtils.generateToken(
      payload,
      this.jwtService,
      this.configService,
    );

    await Promise.all([
      this.userRepository.save(user),
      this.setUserRedis(payload),
      this.userRepository.update(
        { id: user.id },
        { ...token, lastLogin: Utils.currentUtcDatetime() },
      ),
    ]);

    return { ...token, isVerified: user.isVerified };
  }

  async resendCode(body: ResendCodeDto) {
    // check email exist
    const user = await this.userRepository.findOne({
      where: {
        email: body.email.trim(),
        isVerified: false,
        isDeleted: false,
      },
    });
    if (!user)
      throw ApiError.error('Email not exist', ERROR_CODE.EMAIL_NOT_REGISTER, {
        fiell: 'email',
      });

    // generate code
    const code = AuthUtils.generateCode(6);
    user.otpCode = code;
    user.otpCodeCreatedAt = new Date();
    await this.userRepository.save(user);

    // send email
    this.mailService.sendVerifiedRegisterCode(body.email, code);
    return ApiOK.success({ result: true });
  }

  async loginEmail(body: LoginEmailDto) {
    const user = await this.userRepository.findOne({
      where: {
        email: body.email.trim(),
        isDeleted: false,
      },
    });

    if (!user)
      throw ApiError.error(
        'Email has not been registed to the system',
        ERROR_CODE.EMAIL_NOT_REGISTER,
        {
          fiell: 'email',
        },
      );

    const verifyPassword = await AuthUtils.compareData(
      user.password,
      body.password,
    );
    if (!verifyPassword)
      throw ApiError.error('Incorrect password.', ERROR_CODE.WRONG_PASSWORD);

    if (user.isVerified) {
      const payload = {
        id: user.id,
        email: user.email,
        role: user.role,
      } as UserTokenPayload;
      const token = await AuthUtils.generateToken(
        payload,
        this.jwtService,
        this.configService,
      );

      await Promise.all([
        this.setUserRedis(payload),
        this.userRepository.update(
          { id: user.id },
          { ...token, lastLogin: Utils.currentUtcDatetime() },
        ),
      ]);

      return { ...token, isVerified: user.isVerified };
    } else {
      // user not verify, send code for verify
      const code = AuthUtils.generateCode(6);
      user.otpCode = code;
      user.otpCodeCreatedAt = new Date();
      await this.userRepository.save(user);

      this.mailService.sendVerifiedRegisterCode(body.email, code);
      return ApiOK.success({ result: true, isVerified: false });
    }
  }

  async handleLoginApp(body: LoginAppDto) {
    const { accessToken, type, tokenSecret } = body;
    let info: any;
    try {
      info = await this.getInforLoginApp(accessToken, type, tokenSecret);
    } catch (err) {
      this.logger.error(`handleLoginApp() error: ${err}`);
      throw ApiError.error(
        'Cannot login with 3rd party',
        ERROR_CODE.FAILED_LOGIN_3RD,
      );
    }

    let user;
      const [userById, userByEmail] = await Promise.all([
        this.userRepository.findOne({
          where: {
            providerId: info['id'],
          },
        }),
        this.userRepository.findOne({
          where: { email: info['email'] },
        }),
      ]);

    if (!userById) {
      if (!userByEmail) {
        // create new user
        user = await this.userRepository.save({
          email: info['email'],
          name: info['name'].replace(/ /g, '') || '',
          isVerified: true,
          providerId: info['id'],
          status: UserStatus.ACTIVE,
        });
      } else {
        // update user
        user = await this.userRepository.save({
          name: userByEmail.name || info['name'].replace(/ /g, ''),
          providerId: info['id'],
          otpCode: null,
          otpCodeCreatedAt: null,
          isVerified: true,
          status:
            userByEmail.status === UserStatus.DRAFT
              ? UserStatus.ACTIVE
              : userByEmail.status,
        });
      }
    } else {
      user = userById;
    }

    const payload = {
      id: user.id,
      email: user.email,
      role: user.role,
    } as UserTokenPayload;

    const token = await AuthUtils.generateToken(
      payload,
      this.jwtService,
      this.configService,
    );
    await Promise.all([
      this.userRepository.update(
        { id: user.id },
        {
          ...token,
          lastLogin: Utils.currentUtcDatetime(),
        },
      ),
      this.setUserRedis(payload),
    ]);
    return { ...token, isVerified: user.isVerified };
  }

  async setUserRedis(user: UserTokenPayload) {
    await this.redisService.set(`${REDIS_KEY.USER_INFO}-${user.id}`, user, 0);
  }

  async getInforLoginApp(
    accessToken: string,
    type: string,
    tokenSecret?: string,
  ): Promise<AxiosResponse<{ id: string; email: string }>> {
    let result;

    switch (type) {
      case UserAuthenType.GOOGLE:
        result = await this.httpService.axiosRef.get(
          APP_CONFIG.GOOGLE_USER_PROFILE_URL,
          {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          },
        );
        result = result?.data;
        return { ...result, avatar: result?.picture, id: result?.sub };
      default:
        throw ApiError.error('Wrong app type');
    }
  }

  async forgotPassword(body: ForgotPasswordDto) {
    const user = await this.userRepository.findOne({
      where: {
        email: body.email,
      },
    });
    if (!user)
      throw ApiError.error('Email not exist', ERROR_CODE.EMAIL_NOT_REGISTER);

    // generate code for reset
    const code = AuthUtils.generateCode(6);

    user.resetCode = code;
    user.resetCodeCreatedAt = new Date();
    await this.userRepository.save(user);

    this.mailService.sendForgotPasswordConfirmation(body.email, code);

    return ApiOK.success({ result: true });
  }

  async verifyResetPassword(body: VerifyEmailRegisterDto) {
    // check email exist
    const user = await this.userRepository.findOne({
      where: {
        email: body.email.trim(),
        isDeleted: false,
      },
    });
    if (!user)
      throw ApiError.error('Email not exist', ERROR_CODE.EMAIL_NOT_REGISTER, {
        fiell: 'email',
      });

    if (user.resetCode !== body.code)
      throw ApiError.error(
        'Code reset password (1) invalid.',
        ERROR_CODE.CODE_INVALID,
      );

    if (
      Utils.addDateTime(
        user.resetCodeCreatedAt,
        APP_CONFIG.OTP_SETTING.EXPIRED,
      ) < Utils.utcUnixDateTime()
    )
      throw ApiError.error('Code expired', ERROR_CODE.CODE_INVALID);

    const code = AuthUtils.generateCode(6);
    user.resetCodeCreatedAt = null;
    user.resetCode = code;

    await this.userRepository.save(user);
    return ApiOK.success({ result: true, code });
  }

  async resetPassword(body: ResetPasswordDto) {
    const user = await this.userRepository.findOne({
      where: { email: body.email.trim(), isDeleted: false },
    });
    if (!user)
      throw ApiError.error('Email not exist', ERROR_CODE.EMAIL_NOT_REGISTER, {
        fiell: 'email',
      });
    if (user.resetCode !== body.code)
      throw ApiError.error(
        'Code reset password (2) invalid.',
        ERROR_CODE.CODE_INVALID,
      );

    user.resetCode = null;
    user.password = await AuthUtils.hashData(body.password);
    await this.userRepository.save(user);

    return ApiOK.success({ result: true });
  }

  async changePassword(body: ChangePasswordDto, user: User) {
    const userRecord = await this.userRepository.findOne({
      where: { id: user.id, isDeleted: false },
      select: ['id', 'password'],
    });

    if (!userRecord || !userRecord.password)
      throw ApiError.error('User not found or no password set', ERROR_CODE.EMAIL_NOT_REGISTER);

    const verifyCurrentPassword = await AuthUtils.compareData(
      userRecord.password,
      body.currentPassword,
    );
    if (!verifyCurrentPassword)
      throw ApiError.error('Current password is incorrect', ERROR_CODE.WRONG_PASSWORD);

    userRecord.password = await AuthUtils.hashData(body.newPassword);
    await this.userRepository.save(userRecord);

    return ApiOK.success({ result: true });
  }
  

  async getUserInfo(user: User) {
    const userInfo = await this.userRepository.findOne({
      where: { id: user.id },
      
    });
    return userInfo;
  }
}
