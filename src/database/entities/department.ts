import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  OneToMany,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';
import { Major } from './major';
import { Student } from './students';
import { Statistic } from './statistic';


@Entity({ name: 'departments' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Department {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256 })
  name: string;

  @CreateDateColumn()
  createdAt: Date;

  @OneToMany(() => Major, (major) => major.department)
  majors: Major[];

  @OneToMany(() => Student, (student) => student.department)
  students: Student[];

  @OneToMany(() => Statistic, (statistic) => statistic.department)
  statistic: Statistic[];

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
