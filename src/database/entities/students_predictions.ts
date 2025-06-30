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
import { Student } from './students';
import { Semester } from './semester';


@Entity({ name: 'students_predictions' })
@Index(['id', 'createdAt', 'isDeleted'])
export class StudentPrediction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Student, (student) => student.studentPredictions)
  @JoinColumn({ name: 'studentId' })
  student: Student;

  @ManyToOne(() => Semester, (semester) => semester.studentPredictions)
  @JoinColumn({ name: 'semesterId' })
  semester: Semester;

  @CreateDateColumn()
  createdAt: Date;

  @Column({ type: 'float' })
  nextSemesterGPA: number;

  @Column({ type: 'float' })
  nextSemesterCredit: number;

  @Column({ type: 'float' })
  nextSemesterCPA: number;

  @Column({ type: 'float' })
  nextSemesterWarningLevel: number;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
  
}
