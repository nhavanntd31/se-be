export class ApiOK {
  meta: {
    code: number;
    msg?: string;
  };
  data: any;

  constructor(data?: any) {
    this.meta = {
      code: 0,
    };
    this.data = data;
  }

  static success(data: any) {
    return new ApiOK(data);
  }
}
