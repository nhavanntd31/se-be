  import {
    Column,
    CreateDateColumn,
    Entity,
    Index,
    JoinColumn,
    ManyToOne,
    OneToMany,
    PrimaryGeneratedColumn,
    Unique,
    UpdateDateColumn,
  } from 'typeorm';
  import { Department } from './department';
  import { Class } from './class';
  import { StudentProcess } from './student_process';
  import { Major } from './major';
  import { Semester } from './semester';

  @Entity({ name: 'statistics' })
  // @Index(['id', 'createdAt', 'isDeleted'])
  export class Statistic {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column({ length: 256, nullable: true })
    name: string;

    @CreateDateColumn()
    createdAt: Date;

    @ManyToOne(() => Major, (major) => major.statistic)
    @JoinColumn({ name: 'majorId' })
    major: Major;

    @Column({ nullable: true })
    majorId: string;

    @ManyToOne(() => Class, (cls) => cls.statistic)
    @JoinColumn({ name: 'classId' })
    class: Class;

    @Column({ nullable: true })
    classId: string;

    @ManyToOne(() => Department, (department) => department.statistic)
    @JoinColumn({ name: 'departmentId' })
    department: Department;

    @Column({ nullable: true })
    departmentId: string;
    @UpdateDateColumn({ select: false })
    updatedAt: Date;

    @Column({ type: 'float', precision: 10, scale: 2 })
    averageCPA: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    averageGPA: number;

    @Column({ type: 'int' })
    totalStudents: number;

    @Column({ type: 'int' })
    totalStudentIn: number;
    
    @Column({ type: 'int' })
    totalStudentOut: number;

    @Column({ type: 'float' })
    studentGraduationOnTimeRate: number;

    @Column({ type: 'float' })
    studentUngraduationRate: number;
    
    @Column({ type: 'float' })
    studentGraduationLateRate: number;

    @Column({ type: 'float' })
    studentInWarningRate: number;

    @Column({ type: 'int', nullable: true })
    studentGraduationNumber: number;

    @Column({ type: 'int', nullable: true })
    studentUngraduationNumber: number;

    @Column({ type: 'int' })
    studentGraduationLateNumber: number;

    @Column({ type: 'int' })
    studentInWarningNumber: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentWarningOneRate: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentWarningTwoRate: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentWarningThreeRate: number;
    
    @Column({ type: 'float', precision: 10, scale: 2 })
    studentExcellentRate: number;
    
    @Column({ type: 'float', precision: 10, scale: 2 })
    studentVeryGoodRate: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentGoodRate: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentMediumRate: number;

    @Column({ type: 'float', precision: 10, scale: 2 })
    studentBadRate: number;

    
    @ManyToOne(() => Semester, (semester) => semester.statistic)
    @JoinColumn({ name: 'semesterId' })
    semester: Semester;

    @Column()
    semesterId: string;

    @Column({ default: false, select: false })
    isDeleted: boolean;
  }
