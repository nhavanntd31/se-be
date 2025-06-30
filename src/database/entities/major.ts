import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  ManyToOne,
  JoinColumn,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
  OneToMany,
} from 'typeorm';
import { Department } from './department';
import { Class } from './class';
import { Student } from './students';
import { Statistic } from './statistic';


@Entity({ name: 'majors' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Major {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256 })
  name: string;

  @CreateDateColumn()
  createdAt: Date;

  @OneToMany(() => Class, (cls) => cls.major)
  classes: Class[];

  @OneToMany(() => Student, (student) => student.major)
  students: Student[];

  @OneToMany(() => Statistic, (statistic) => statistic.major)
  statistic: Statistic[];

  @ManyToOne(() => Department, (department) => department.majors)
  @JoinColumn({ name: 'departmentId' })
  department: Department;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
