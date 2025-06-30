import { Body, Controller, Get, HttpCode, HttpStatus, Post, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags, ApiUnauthorizedResponse } from '@nestjs/swagger';
import { AuthService } from './auth.service';
import {
  LoginEmailDto,
  RegisterEmailDto,
  ForgotPasswordDto,
  ResetPasswordDto,
  ResendCodeDto,
  VerifyEmailRegisterDto,
  LoginAppDto,
  ChangePasswordDto,
} from './dto';
import { JwtAuthGuard } from 'src/common/guards/jwt-auth.guard';
import { CurrentUser } from 'src/common/decorators/user.decorator';
import { PublicGuard } from 'src/common/guards';
import { User } from 'src/database/entities';

@ApiTags('Auth')
@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @Post('register')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Register by email' })
  async registerByEmail(@Body() body: RegisterEmailDto) {
    return this.authService.registerByEmail(body);
  }

  @Post('verify')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Register by email' })
  async verifyRegister(@Body() body: VerifyEmailRegisterDto) {
    return this.authService.verifyRegister(body);
  }

  @Post('register/resend-code')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Resend code for register' })
  async resendCode(@Body() body: ResendCodeDto) {
    return this.authService.resendCode(body);
  }

  @Post('login')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Login' })
  async loginEmail(@Body() body: LoginEmailDto) {
    return this.authService.loginEmail(body);
  }

  @Post('login/app')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Login 3rd party' })
  async appLogin(@Body() body: LoginAppDto) {
    return this.authService.handleLoginApp(body);
  }

  @Post('forgot-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Forgot password' })
  async forgotPassword(@Body() body: ForgotPasswordDto) {
    return this.authService.forgotPassword(body);
  }

  @Post('verify/reset-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Verify code reset password' })
  async verifyResetPassword(@Body() body: VerifyEmailRegisterDto) {
    return this.authService.verifyResetPassword(body);
  }

  @Post('reset-password')
  @HttpCode(HttpStatus.OK)
  async resetPassword(@Body() body: ResetPasswordDto) {
    return this.authService.resetPassword(body);
  }

  @Post('change-password')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @UseGuards(PublicGuard)
  @ApiUnauthorizedResponse({ description: 'Unauthorized' })
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ description: 'Change password for authenticated user' })
  async changePassword(@Body() body: ChangePasswordDto, @CurrentUser() user: User) {
    return this.authService.changePassword(body, user);
  }

  @Get('user-info')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @UseGuards(PublicGuard)
  @ApiUnauthorizedResponse({ description: 'Unauthorized' })
  @HttpCode(HttpStatus.OK)
  async getUserInfo(@CurrentUser() user: User) {
    return this.authService.getUserInfo(user);
  }
}
