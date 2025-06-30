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
import { Major } from './major';
import { Student } from './students';
import { Statistic } from './statistic';

@Entity({ name: 'classes' })
@Index(['id', 'createdAt', 'isDeleted'])
export class Class {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 256 })
  name: string;

  @CreateDateColumn()
  createdAt: Date;

  @OneToMany(() => Student, (student) => student.class)
  students: Student[];

  @OneToMany(() => Statistic, (statistic) => statistic.class)
  statistic: Statistic[];

  @ManyToOne(() => Major, (major) => major.classes)
  @JoinColumn({ name: 'majorId' })
  major: Major;

  @UpdateDateColumn({ select: false })
  updatedAt: Date;

  @Column({ default: false, select: false })
  isDeleted: boolean;
}
