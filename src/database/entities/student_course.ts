import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  JoinColumn,
  ManyToOne,
  PrimaryGeneratedColumn,
  Unique,
  UpdateDateColumn,
} from 'typeorm';
import { Semester } from './semester';
import { Student } from './students';

@Entity({ name: 'student_courses' })
@Index(['id', 'createdAt', 'isDeleted'])
@Unique(['studentId', 'semesterId', 'courseId'])
export class StudentCourse {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Semester, (semester) => semester.studentCourses)
  @JoinColumn({ name: 'semesterId' })
  semester: Semester;

  @Column({})
  semesterId: string;

  @Column({ type: 'varchar', length: 50 })
  courseId: string;

  @Column({ type: 'varchar', length: 256 })
  courseName: string;

  @Column({ type: 'int' })
  credits: number;

  @Column({ type: 'varchar', length: 50 })
  class: string;

  @Column({ type: 'float', nullable: true })
  continuousAssessmentScore: number;

  @Column({ type: 'float', nullable: true })
  examScore: number;

  @Column({ type: 'varchar', length: 2, nullable: true })
  finalGrade: string;

  @Column({ type: 'int' })
  relativeTerm: number;

  @ManyToOne(() => Student, (student) => student.studentCourses)
  @JoinColumn({ name: 'studentId' })
  student: Student;

  @Column({})
  studentId: string;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
