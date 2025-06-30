import { Injectable } from '@nestjs/common';
import { S3 } from 'aws-sdk';
import * as mime from 'mime-types';
import { ApiError } from 'src/common/responses';
import { Utils } from 'src/common/utils/utils';
import { ConfigService } from 'src/config/config.service';

@Injectable()
export class S3Service {
  constructor(private readonly configService: ConfigService) {}

  private s3 = this.configService.s3Config.accessKey
    ? new S3({
        credentials: {
          accessKeyId: this.configService.s3Config.accessKey,
          secretAccessKey: this.configService.s3Config.sceret,
        },
      })
    : new S3();

  async getPreSignUrl(file: Express.Multer.File, folder?: string) {
    const type = mime.contentType(file.originalname);
    const key = `${folder ? folder + '/' : ''}${Utils.valueOfDateTime()}_${
      file.originalname
    }`;
    const presignUrl = await this.s3.getSignedUrl('putObject', {
      Bucket: this.configService.s3Config.bucket,
      Key: key,
      Expires: 5 * 60,
      //   ACL: 'public-read',
      ContentType: type,
    });

    return { presignUrl, type, key };
  }

  async uploadS3(folder: string, buffer, filename) {
    const key = `${
      folder ? folder + '/' : ''
    }${Utils.valueOfDateTime()}_${filename}`;
    try {
      const result = await this.s3
        .upload({
          Bucket: this.configService.s3Config.bucket,
          Key: key,
          Body: buffer,
        })
        .promise();
      return result;
    } catch (e) {
      console.error('uploadS3::error', e);
      throw ApiError.error('uploadS3::error', e);
    }
  }
}
