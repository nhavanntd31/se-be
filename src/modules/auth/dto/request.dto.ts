import { ApiProperty } from '@nestjs/swagger';
import {
  IsEmail,
  IsNotEmpty,
  IsOptional,
  IsStrongPassword,
  MaxLength,
  Validate,
} from 'class-validator';
import { UsernameValidationRule } from 'src/common/validations';
import { ERROR_CODE } from 'src/common/constants';

export class RegisterEmailDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  name: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsStrongPassword(
    {
      minLength: 8,
      minNumbers: 1,
      minSymbols: 1,
      minUppercase: 1,
      minLowercase: 1,
    },
    { message: ERROR_CODE.PASSWORD_FORMAT },
  )
  @MaxLength(20, { message: ERROR_CODE.MAX_LENGTH })
  password: string;
}

export class VerifyEmailRegisterDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @MaxLength(6, { message: ERROR_CODE.MAX_LENGTH })
  code: string;
}

export class ResendCodeDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;
}

export class LoginEmailDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsStrongPassword(
    {
      minLength: 8,
      minNumbers: 1,
      minSymbols: 1,
      minUppercase: 1,
      minLowercase: 1,
    },
    { message: ERROR_CODE.PASSWORD_FORMAT },
  )
  @MaxLength(20, { message: ERROR_CODE.MAX_LENGTH })
  password: string;
}

export class LoginAppDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  accessToken: string;

  @ApiProperty({
    description: `google = 'GOOGLE' ...`,
  })
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  type: string;

  @ApiProperty({ required: false })
  @IsOptional()
  tokenSecret?: string;
}

export class ForgotPasswordDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;
}

export class ResetPasswordDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsEmail({}, { message: ERROR_CODE.EMAIL_FORMAT })
  @MaxLength(256, { message: ERROR_CODE.MAX_LENGTH })
  email: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  code: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsStrongPassword(
    {
      minLength: 8,
      minNumbers: 1,
      minSymbols: 1,
      minUppercase: 1,
      minLowercase: 1,
    },
    { message: ERROR_CODE.PASSWORD_FORMAT },
  )
  @MaxLength(20, { message: ERROR_CODE.MAX_LENGTH })
  password: string;
}

export class ChangePasswordDto {
  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsStrongPassword(
    {
      minLength: 8,
      minNumbers: 1,
      minSymbols: 1,
      minUppercase: 1,
      minLowercase: 1,
    },
    { message: ERROR_CODE.PASSWORD_FORMAT },
  )
  @MaxLength(20, { message: ERROR_CODE.MAX_LENGTH })
  currentPassword: string;

  @ApiProperty()
  @IsNotEmpty({ message: ERROR_CODE.REQUIRED })
  @IsStrongPassword(
    {
      minLength: 8,
      minNumbers: 1,
      minSymbols: 1,
      minUppercase: 1,
      minLowercase: 1,
    },
    { message: ERROR_CODE.PASSWORD_FORMAT },
  )
  @MaxLength(20, { message: ERROR_CODE.MAX_LENGTH })
  newPassword: string;
}
