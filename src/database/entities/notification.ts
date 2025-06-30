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
import { NotificationUser } from './notification_user';

@Entity({ name: 'notifications' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Notification {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'text', nullable: true })
  title: string;

  @Column({ type: 'text', nullable: true })
  content: string;

  @Column({ type: 'text', nullable: true })
  link: string;

  @Column({ default: false })
  isRead: boolean;

  @OneToMany(() => NotificationUser, (notificationUser) => notificationUser.notification)
  notificationUsers: NotificationUser[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
