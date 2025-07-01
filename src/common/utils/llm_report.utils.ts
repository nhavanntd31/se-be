import * as fs from 'fs'
import * as path from 'path'
import * as puppeteer from 'puppeteer'
import { GoogleGenerativeAI } from '@google/generative-ai'
import { StudentProcess } from 'src/database/entities/student_process'
import { Student } from 'src/database/entities/students'

const GOOGLE_API_KEY = "AIzaSyDzyARw95GU-P6HHfAKdAicnQNLWWQHC18"
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY)

const promptTemplate = `Dưới đây là thông tin học tập của một sinh viên:

{text}

Hãy điền thông tin vào template HTML sau đây, thay thế các placeholder bằng nội dung phù hợp:

<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo đánh giá học tập sinh viên</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #F8F9FA; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }
        .header { text-align: center; color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }
        .student-info { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 25px; }
        .student-info table { width: 100%; border-collapse: collapse; }
        .student-info td { padding: 8px; border-bottom: 1px solid #ddd; }
        .student-info td:first-child { font-weight: bold; color: #2E86AB; width: 30%; }
        .performance-chart { background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 25px; }
        .section { margin-bottom: 25px; }
        .section h3 { color: #2E86AB; border-left: 4px solid #2E86AB; padding-left: 15px; }
        .section ul { padding-left: 20px; }
        .section li { margin-bottom: 8px; line-height: 1.6; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }
        @media print { body { background: white; } .container { box-shadow: none; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BÁO CÁO ĐÁNH GIÁ HỌC TẬP SINH VIÊN</h1>
        </div>
        
        <div class="student-info">
            <table>
                <tr><td>Mã sinh viên:</td><td>{student_id}</td></tr>
                <tr><td>Họ tên:</td><td>{student_name}</td></tr>
                <tr><td>Lớp:</td><td>{class_name}</td></tr>
                <tr><td>Ngành:</td><td>{major_name}</td></tr>
                <tr><td>Khoa:</td><td>{department_name}</td></tr>
                <tr><td>Số học kỳ đã hoàn thành:</td><td>{semester_count}</td></tr>
            </table>
        </div>

        <div class="performance-chart">
            <h3>Biểu đồ xu hướng điểm số</h3>
            <pre>{performance_chart}</pre>
        </div>

        <div class="section">
            <h3>ĐÁNH GIÁ KẾT QUẢ HỌC TẬP</h3>
            <ul>
                {evaluation_points}
            </ul>
        </div>

        <div class="section">
            <h3>ĐỀ XUẤT HƯỚNG PHÁT TRIỂN</h3>
            <ul>
                {recommendation_points}
            </ul>
        </div>

        <div class="footer">
            <p>Báo cáo được tạo ngày: {current_date}</p>
            <p><strong>Cố vấn học tập</strong></p>
        </div>
    </div>
</body>
</html>

Yêu cầu:
1. Thay thế {student_id}, {student_name}, {class_name}, {major_name}, {department_name}, {semester_count} bằng thông tin thực tế
2. Tạo {performance_chart} dạng text-based chart hiển thị xu hướng CPA qua các học kỳ
3. Tạo {evaluation_points} với 3-4 bullet points đánh giá kết quả học tập
4. Tạo {recommendation_points} với 3-4 bullet points đề xuất cải thiện
5. Thay {current_date} bằng ngày hiện tại
6. Chỉ trả về HTML hoàn chỉnh, không thêm text nào khác`

function buildStudentReportText(student: Student, processes: StudentProcess[]) {
  const info = [
    `Mã sinh viên: ${student.studentId}`,
    `Họ tên: ${student.name}`,
    `Lớp: ${student.class?.name || ''}`,
    `Ngành: ${student.major?.name || ''}`,
    `Khoa: ${student.department?.name || ''}`,
    `Số học kỳ đã hoàn thành: ${processes.length}`
  ]
  const processRows = processes
    .sort((a, b) => a.semester.name.localeCompare(b.semester.name))
    .map(p => `| ${p.semester.name} | ${p.cpa} | ${p.registeredCredits} | ${p.debtCredits} | ${p.warningLevel} |`)
  const processTable = [
    '| Học kỳ | CPA | Tín chỉ đăng ký | Tín chỉ nợ | Cảnh báo |',
    '|--------|-----|----------------|------------|----------|',
    ...processRows
  ].join('\n')
  return info.join('\n') + '\n' + processTable
}

async function generateStudentReportHTML(student: any, processes: any[]) {
  const studentText = buildStudentReportText(student, processes)
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' })
  const result = await model.generateContent({ contents: [{ role: 'user', parts: [{ text: promptTemplate.replace('{text}', studentText) }] }] })
  const html = result.response.candidates?.[0]?.content?.parts?.[0]?.text || ''
  if (!html.includes('<html')) throw new Error('LLM did not return valid HTML')
  return html
}

async function htmlToPdfBuffer(html: string): Promise<Buffer> {
  const browser = await puppeteer.launch({ headless: true })
  const page = await browser.newPage()
  await page.setContent(html, { waitUntil: 'networkidle0' })
  const uint8Array = await page.pdf({ format: 'A4', printBackground: true })
  await browser.close()
  return Buffer.from(uint8Array)
}

export async function generateStudentPDFReport(student: any, processes: any[]): Promise<Buffer> {
  const html = await generateStudentReportHTML(student, processes)
  const buffer = await htmlToPdfBuffer(html)
  return buffer
} 