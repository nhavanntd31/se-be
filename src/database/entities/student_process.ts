import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  JoinColumn,
  ManyToOne,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
  Unique,
} from 'typeorm';
import { Semester } from './semester';
import { Student } from './students';

@Entity({ name: 'student_processes' })
@Index(['id', 'createdAt', 'isDeleted'])
@Unique(['semesterId', 'studentId'])
export class StudentProcess {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Student, (student) => student.studentProcesses)
  @JoinColumn({ name: 'studentId' })
  student: Student;

  @Column()
  studentId: string;

  @ManyToOne(() => Semester, (semester) => semester.studentProcesses)
  @JoinColumn({ name: 'semesterId' })
  semester: Semester;

  @Column()
  semesterId: string;

  @Column({ type: 'float' })
  gpa: number;

  @Column({ type: 'float' })
  cpa: number;

  @Column({ type: 'int' })
  registeredCredits: number;

  @Column({ type: 'int' })
  passedCredits: number;
  
  @Column({ type: 'int' })
  debtCredits: number;

  @Column({ type: 'int' })
  totalAcceptedCredits: number;

  @Column({ type: 'int' })
  warningLevel: number;

  @Column({ type: 'int' })
  studentLevel: number;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
