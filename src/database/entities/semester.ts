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
import { StudentProcess } from './student_process';
import { StudentCourse } from './student_course';
import { Statistic } from './statistic';
import { StudentPrediction } from './students_predictions';


@Entity({ name: 'semesters' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Semester {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256, unique: true })
  name: string;

  @OneToMany(() => StudentProcess, (studentProcess) => studentProcess.semester)
  studentProcesses: StudentProcess[];

  @OneToMany(() => StudentCourse, (studentCourse) => studentCourse.semester)
  studentCourses: StudentCourse[];

  @OneToMany(() => Statistic, (statistic) => statistic.semester)
  statistic: Statistic[];

  @OneToMany(() => StudentPrediction, (studentPrediction) => studentPrediction.semester)
  studentPredictions: StudentPrediction[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
