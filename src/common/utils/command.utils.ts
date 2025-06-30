import { exec } from 'child_process';

export class CommandUtils {
  public static async runCommand(cmd: string) {
    const result = await new Promise((resolve, reject) => {
      exec(cmd, (err, out, outerr) => {
        if (err) {
          reject(err);
        }
        resolve(out);
      });
    });
    return result;
  }
}
