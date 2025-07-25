import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  OneToMany,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';
import { NotificationUser } from './notification_user';

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
}

export enum UserStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  DEACTIVE = 'deactive',
}

export enum UserAuthenType {
  EMAIL = 'email',
  GOOGLE = 'google',
}

export interface UserTokenPayload {
  id: string;
  email: string;
  role: string;
}

export enum UserPermission {
  USER_ALL = 'user_all',
  USER_DEPARTMENT = 'user_department',
  USER_MAJOR = 'user_major',
  USER_CLASS = 'user_class',
  USER_STUDENT = 'user_student',
  USER_PCLO = 'user_pclo',
}

@Entity({ name: 'users' })
@Index(['id', 'createdAt', 'isDeleted'])
export class User {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256 })
  name: string;

  @Column({ length: 256, nullable: true })
  password: string;

  @Column({ length: 20, default: UserRole.USER })
  role: string;
    
  @Column({ length: 50, default: UserAuthenType.EMAIL })
  authenType: string;

  @Column({ length: 100, unique: true, nullable: true })
  providerId: string;

  @Column({ length: 20, default: UserStatus.DRAFT })
  status: string;

  @Column({ length: 20, default: UserPermission.USER_ALL })
  permission: string;

  @Column({ nullable: false, length: 256, unique: true })
  email: string;

  @Column({ nullable: true, length: 6 })
  otpCode: string;

  @Column({ nullable: true, type: 'timestamp' })
  otpCodeCreatedAt: Date;

  @Column({ nullable: true, length: 6 })
  resetCode: string;

  @Column({ nullable: true, type: 'timestamp' })
  resetCodeCreatedAt: Date;

  @Column({ default: false })
  isVerified: boolean;

  @Column({ nullable: true, select: false })
  accessToken: string;

  @Column({ nullable: true, select: false })
  refreshToken: string;

  @Column({ nullable: true, type: 'timestamp' })
  lastLogin: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;

  @OneToMany(() => NotificationUser, (notificationUser) => notificationUser.user)
  notificationUsers: NotificationUser[];
}
