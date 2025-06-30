import { IsNotEmpty, IsOptional, IsArray, IsNumber, Min, Max } from "class-validator";
import { IsString } from "class-validator";
import { Type } from "class-transformer";
import { BaseSearchDto } from "src/common/constants";

export class GetStatisticInfoDto {
  @IsString()
  @IsNotEmpty()
  semesterId: string;

  @IsString()
  @IsOptional()
  departmentId?: string;

  @IsString()   
  @IsOptional()
  majorId?: string;

  @IsString()
  @IsOptional()
  classId?: string;
}

export class GetCPATrajectoryDto {
  @IsString()
  @IsOptional()
  departmentId?: string;

  @IsString()   
  @IsOptional()
  majorId?: string;

  @IsString()
  @IsOptional()
  classId?: string;

  @IsString()
  @IsNotEmpty()
  startSemester: string;

  @IsString()
  @IsNotEmpty()
  endSemester: string;

  @IsArray()
  @IsNumber({}, { each: true })
  @Min(0, { each: true })
  @Max(100, { each: true })
  @Type(() => Number)
  thresholdRates: number[];

  @IsArray()
  @IsString({ each: true })
  @IsOptional()
  studentIds?: string[];
}

export class GetStudentsBySemesterRangeDto extends BaseSearchDto {
  @IsString()
  @IsOptional()
  departmentId?: string;

  @IsString()   
  @IsOptional()
  majorId?: string;

  @IsString()
  @IsOptional()
  classId?: string;

  @IsString()
  @IsNotEmpty()
  startSemester: string;

  @IsString()
  @IsNotEmpty()
  endSemester: string;
}

