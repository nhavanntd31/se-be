import * as moment from 'moment';

export class Utils {
  public static getUniqueArray(data: any[]) {
    return data.filter((v, i, a) => a.indexOf(v) === i);
  }

  public static replaceText(text, args) {
    let str = text.toString();
    for (const key in args) {
      str = str.replace(new RegExp('\\{' + key + '\\}', 'gi'), args[key]);
    }
    return str;
  }

  public static escapeRegex(string: string) {
    return string.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
  }

  public static stripHtml(html: string) {
    if (!html) {
      return;
    }
    const result = html.replace(/<[^>]*>?/gm, '').replace(/&nbsp;/gi, ' ');
    return result;
  }

  public static currentUtcDatetime() {
    return moment().utc().toDate();
  }

  public static utcUnixDateTime() {
    return moment().utc().unix();
  }

  public static unixDateTime() {
    return moment().unix();
  }

  public static valueOfDateTime() {
    return moment().valueOf();
  }

  public static valueOfUtcDateTime() {
    return moment().utc().valueOf();
  }

  public static addDateTime(date, timeAdd = 0, typeAdd = 'seconds') {
    const addType = typeAdd as moment.unitOfTime.DurationConstructor;
    return moment(date ? date : new Date())
      .add(timeAdd, addType)
      .unix();
  }

  public static subtractDateTime(
    date,
    subtractTime = 0,
    typeSubtract = 'seconds',
  ) {
    const subType = typeSubtract as moment.unitOfTime.DurationConstructor;
    return moment(date ? date : new Date())
      .subtract(subtractTime, subType)
      .utc()
      .toDate();
  }

  public static addDateTimeValueOf(date, timeAdd = 0, typeAdd = 'seconds') {
    const addType = typeAdd as moment.unitOfTime.DurationConstructor;
    return moment(date ? date : new Date())
      .add(timeAdd, addType)
      .valueOf();
  }

  public static subtractTime(
    day: any,
    subtractNumber: number,
    typeSub = 'day',
    formatTime = 'YYYY-MM-DD HH:mm:ss',
  ) {
    const subType = typeSub as moment.unitOfTime.DurationConstructor;
    return moment(day ? day : new Date())
      .subtract(subtractNumber, subType)
      .startOf(subType)
      .format(formatTime);
  }

  public static startOfDay(
    date: any,
    typeSub = 'day',
    formatTime = 'YYYY-MM-DD HH:mm:ss',
  ) {
    const subType = typeSub as moment.unitOfTime.DurationConstructor;
    return moment(date ? date : new Date())
      .startOf(subType)
      .format(formatTime);
  }

  public static endOfDay(date: string, formatTime = 'YYYY-MM-DD HH:mm:ss') {
    return moment(date ? date : new Date())
      .endOf('day')
      .format(formatTime);
  }

  public static formatToUtc(date) {
    return moment(date ? date : new Date()).unix();
  }
  public static toFloat(number: number, precision: number) {
    return Math.round(number * Math.pow(10, precision)) / Math.pow(10, precision);
  }
}

export const generateSemesterList = (startSemester: string, endSemester: string): string[] => {
  const semesters: string[] = [];
  
  const parseYear = (semester: string) => parseInt(semester.substring(0, 4));
  const parseTerm = (semester: string) => parseInt(semester.substring(4));
  
  let currentYear = parseYear(startSemester);
  let currentTerm = parseTerm(startSemester);
  
  const endYear = parseYear(endSemester);
  const endTerm = parseTerm(endSemester);
  
  while (currentYear < endYear || (currentYear === endYear && currentTerm <= endTerm)) {
    const semesterName = `${currentYear}${currentTerm}`;
    semesters.push(semesterName);
    
    currentTerm++;
    if (currentTerm > 2) {
      currentTerm = 1;
      currentYear++;
    }
  }
  
  return semesters;
};

export const isSemesterInRange = (semester: string, startSemester: string, endSemester: string): boolean => {
  const semesterList = generateSemesterList(startSemester, endSemester);
  return semesterList.includes(semester);
};
