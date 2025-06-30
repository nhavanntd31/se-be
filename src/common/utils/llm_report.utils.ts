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

Hãy tạo báo cáo đánh giá học tập dưới dạng HTML hoàn chỉnh với các yêu cầu sau:
1. Đánh giá sự thay đổi trong kết quả học tập qua từng học kỳ
2. Nhận định về xu hướng học lực hiện tại
3. Đưa ra đề xuất thực tế nhằm duy trì hoặc nâng cao kết quả học tập
4. Thêm nhận xét tổng kết và lưu ý cần thiết
5. Trả lời với tông điệu của cố vấn học tập chuyên nghiệp
Trả về HTML hoàn chỉnh với cấu trúc sau:
- DOCTYPE html với encoding UTF-8
- CSS inline styling chuyên nghiệp (màu xanh dương chủ đạo)
- Header: "BÁO CÁO ĐÁNH GIÁ HỌC TẬP SINH VIÊN"
- Thông tin cơ bản sinh viên trong bảng
- Biểu đồ xu hướng điểm số (text-based chart)
- Phần "ĐÁNH GIÁ KẾT QUẢ HỌC TẬP" với bullet points
- Phần "ĐỀ XUẤT HƯỚNG PHÁT TRIỂN" với action items
- Footer với thời gian tạo báo cáo và chữ ký cố vấn
CSS styling yêu cầu:
- Font chữ Arial/sans-serif
- Màu chủ đạo: #2E86AB (xanh dương)
- Background: #F8F9FA
- Border radius cho các card
- Shadow effects nhẹ
- Spacing hợp lý
- Print-friendly
`

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