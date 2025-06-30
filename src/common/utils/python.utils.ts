import { spawn } from 'child_process';
import { concat } from 'lodash';

export class PythonUtils {
  public static async call(path: string, args?: any[]) {
    const params = args ? concat([path], args) : [path];
    console.log('Python command params:', params);
    
    try {
      const result = await new Promise((resolve, reject) => {
        const pyModule = spawn('python', params);
        let stdout = '';
        let stderr = '';

        pyModule.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        pyModule.stderr.on('data', (data) => {
          stderr += data.toString();
          console.error('Python stderr:', data.toString());
        });

        pyModule.on('close', (code) => {
          if (code === 0) {
            resolve(stdout);
          } else {
            reject(new Error(`Python process exited with code ${code}. Stderr: ${stderr}`));
          }
        });

        pyModule.on('error', (error) => {
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });
      });

      return result.toString();
    } catch (error) {
      console.error('Python execution error:', error);
      throw error;
    }
  }
}