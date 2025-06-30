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
import { Department } from './department';
import { Class } from './class';
import { StudentProcess } from './student_process';
import { Major } from './major';
import { StudentCourse } from './student_course';
import { StudentPrediction } from './students_predictions';

@Entity({ name: 'upload_events' })
@Index(['id', 'createdAt', 'isDeleted'])
export class UploadEvent {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'text' })
  courseFilePath: string;

  @Column({ type: 'text' })
  performanceFilePath: string;

  @CreateDateColumn()
  createdAt: Date;

  @Column({ default: false })
  isImportSuccess: boolean;

  @Column({ type: 'timestamp', nullable: true })
  importStartedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  importCompletedAt: Date;

  @Column({ type: 'text', nullable: true })
  importFailedMessage: string;

  @Column({ default: false })
  isStatisticSuccess: boolean;

  @Column({ type: 'text', nullable: true })
  statisticFailedMessage: string;

  @Column({ type: 'timestamp', nullable: true })
  statisticCompletedAt: Date; 

  @Column({ type: 'timestamp', nullable: true })
  statisticStartedAt: Date;

  @Column({ default: false })
  isPredictSuccess: boolean;

  @Column({ type: 'timestamp', nullable: true })
  predictStartedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  predictCompletedAt: Date;

  @Column({ type: 'text', nullable: true })
  predictFailedMessage: string;

  @Column({ type: 'text', nullable: true }) 
  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
