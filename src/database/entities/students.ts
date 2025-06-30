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

@Entity({ name: 'students' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Student {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256 })
  name: string;

  @Column({ length: 256, unique: true })
  studentId: string;

  @CreateDateColumn()
  createdAt: Date;

  @ManyToOne(() => Major, (major) => major.students)
  @JoinColumn({ name: 'majorId' })
  major: Major;

  @Column({})
  majorId: string;

  @ManyToOne(() => Class, (cls) => cls.students)
  @JoinColumn({ name: 'classId' })
  class: Class;

  @Column({})
  classId: string;

  @ManyToOne(() => Department, (department) => department.students)
  @JoinColumn({ name: 'departmentId' })
  department: Department;

  @Column({})
  departmentId: string;

  @OneToMany(() => StudentCourse, (studentCourse) => studentCourse.student)
  studentCourses: StudentCourse[];

  @OneToMany(() => StudentProcess, (studentProcess) => studentProcess.student)
  studentProcesses: StudentProcess[];

  @OneToMany(
    () => StudentPrediction,
    (studentPrediction) => studentPrediction.student,
  )
  studentPredictions: StudentPrediction[];

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
