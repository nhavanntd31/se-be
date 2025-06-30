import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  JoinColumn,
  ManyToOne,
  OneToMany,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';
import { User } from './user.entity';
import { Notification } from './notification';
@Entity({ name: 'notification_users' })
@Index(['id', 'createdAt', 'isDeleted'])
export class NotificationUser {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Notification, (notification) => notification.notificationUsers)
  @JoinColumn({ name: 'notificationId' })
  notification: Notification;

  @Column({ })
  notificationId: string;

  @ManyToOne(() => User, (user) => user.notificationUsers)
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column({ })
  userId: string;

  @Column({ default: false })
  isRead: boolean;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
