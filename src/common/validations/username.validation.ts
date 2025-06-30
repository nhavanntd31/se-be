import {
  ValidationArguments,
  ValidatorConstraint,
  ValidatorConstraintInterface,
} from 'class-validator';

@ValidatorConstraint({ name: 'UsernameValidation', async: true })
export class UsernameValidationRule implements ValidatorConstraintInterface {
  validate(
    value: any,
    validationArguments?: ValidationArguments,
  ): boolean | Promise<boolean> {
    if (/[^a-zA-Z0-9._-]/g.test(value)) return false;
    return true;
  }

  defaultMessage?(validationArguments?: ValidationArguments): string {
    return `E33`;
  }
}
