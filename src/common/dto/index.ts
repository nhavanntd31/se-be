import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import { IsNotEmpty, IsNumber, IsOptional } from 'class-validator';

export class BaseSearchDto {
  @ApiProperty({ required: false })
  @IsOptional()
  keyword?: string;

  @ApiProperty({ required: false })
  @IsOptional()
  sortField?: string;

  @ApiProperty({ required: false, description: '-1: DESC, 1: ASC' })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  sortType?: number;

  @ApiProperty({ required: false })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  offset?: number;

  @ApiProperty({ required: false })
  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  limit?: number;
}

export class RunCommandDto {
  @ApiProperty({ required: true })
  @IsNotEmpty()
  cmd: string;

  @ApiProperty({ required: true })
  @IsNotEmpty()
  key: string;
}
