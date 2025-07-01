import * as ExcelJS from 'exceljs'
import axios from 'axios'
import * as fs from 'fs'
import * as path from 'path'
import { v4 as uuidv4 } from 'uuid'

function getParam(paramBuffer?: Buffer) {
  let param: any = {}
  if (paramBuffer) {
    try { param = JSON.parse(paramBuffer.toString()) } catch {}
  }
  return param
}

async function askLLMOpenRouter(prompt: string, param: any = {}) {
  const api_key = param.api_key || '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'
  const model = param.model || 'mistralai/mistral-small-3.2-24b-instruct:free'
  const temperature = param.temperature ?? 0.5
  const system_instruction = param.system_instruction || 'You are a helpful and truthful conversational AI.'
  const url = 'https://openrouter.ai/api/v1/chat/completions'
  const headers = {
    'Authorization': `Bearer ${api_key}`,
    'Content-Type': 'application/json'
  }
  const payload: any = {
    model,
    messages: [
      { role: 'system', content: system_instruction },
      { role: 'user', content: prompt }
    ],
    max_tokens: 32000,
    temperature
  }
  if (param.tools) {
    payload.tools = param.tools
    payload.tool_choice = 'auto'
  }
  const response = await axios.post(url, payload, { headers, timeout: 120000 })
  const result = response.data
  return result.choices[0].message.content || ''
}

const DEFAULT_PROMPT = '# Phân tích chuẩn đầu ra chương trình đào tạo (PLO) theo khung phân loại Bloom hai chiều\n\n## Bước 1: Phân tích từng PLO\nVới mỗi PLO được cung cấp, hãy:\n1. **Đánh giá tính đúng quy định**: PLO có rõ ràng, cụ thể, đo lường được và định hướng năng lực không? Có đúng cấu trúc không?\n2. **Xác định nhóm thuộc tính PLO** theo 6 nhóm phổ biến:\n   - Kiến thức chuyên môn\n   - Kỹ năng nghề nghiệp\n   - Năng lực giải quyết vấn đề\n   - Năng lực giao tiếp & làm việc nhóm\n   - Năng lực học tập suốt đời\n   - Năng lực đạo đức & trách nhiệm xã hội\n3. **Ánh xạ PLO vào Thang Bloom hai chiều**:\n   - Chiều tiến trình nhận thức: [Remember, Understand, Apply, Analyze, Evaluate, Create]\n   - Chiều loại kiến thức: [Factual Knowledge, Conceptual Knowledge, Procedural Knowledge, Meta-Cognitive Knowledge]\n\n**Lưu ý**: Tránh các động từ không đo lường được như: Hiểu, Biết, Nắm được, Nhận thấy, Chấp nhận, Có kiến thức về, Nhận thức được, Có ý thức về, Học được, Nhận biết, Hình thành giá trị, Chấp nhận, Làm quen với.\n\n## Bước 2: Trình bày kết quả\n- Viết phân tích chi tiết dưới dạng đoạn văn cho từng PLO, nêu rõ:\n  - PLO có phù hợp và hiệu lực không?\n  - PLO thuộc nhóm năng lực nào?\n  - Lý do lựa chọn mức độ nhận thức và loại kiến thức tương ứng.\n- Mỗi phân tích PLO phải bắt đầu bằng tiêu đề: ### Phân tích: <PLO_ID>\n- Tổng hợp tất cả các PLO vào một bảng ánh xạ Bloom ở cuối dưới dạng bảng Markdown:\n\n| Mã PLO | Mô tả rút gọn | Nhóm Năng lực | Tiến trình nhận thức | Loại kiến thức |\n|--------|---------------|---------------|----------------------|----------------|\n| PLO1   | Mô tả ngắn gọn | | | |\n\n## Yêu cầu định dạng:\n- Đầu ra sử dụng Markdown, có tiêu đề rõ ràng cho từng phần.\n- Phân tích từng PLO theo thứ tự cung cấp, gắn nhãn rõ ràng (PLO01, PLO02, …).\n- Trình bày bảng gọn gàng, dễ đọc.\n- Chỉ trình bày kết quả phân tích, không diễn giải thêm.'

export async function analyzePLOExcel(
  excelBuffer: Buffer,
  paramBuffer?: Buffer
): Promise<{ analyzeBuffer: Buffer, bloomBuffer: Buffer, bloomTable: any[] }> {
  if (!excelBuffer || excelBuffer.length === 0) {
    throw new Error('Excel buffer is empty or invalid')
  }

  const tempDir = path.join(__dirname, '../../..', 'temp')
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true })
  }
  
  const tempFile = path.join(tempDir, `temp_${uuidv4()}.xlsx`)
  
  try {
    fs.writeFileSync(tempFile, excelBuffer)
    
    const workbook = new ExcelJS.Workbook()
    await workbook.xlsx.readFile(tempFile)
    const sheet = workbook.worksheets[0]
    const ploData: { id: string, plo: string }[] = []
    sheet.eachRow((row, idx) => {
      if (idx === 1) return
      const id = row.getCell(1).text.trim()
      const plo = row.getCell(2).text.trim()
      if (id && plo) ploData.push({ id, plo })
    })

    const param = getParam(paramBuffer)
    const customPrompt = param.prompt || DEFAULT_PROMPT
    const prompt = `Phân tích các chuẩn đầu ra chương trình đào tạo (PLO) sau đây theo yêu cầu được cung cấp. Thực hiện phân tích cho từng PLO và tổng hợp kết quả vào một bảng ánh xạ Bloom cuối cùng.\n${customPrompt}\n\nDanh sách PLO:\n${ploData.map(p=>`**Mã PLO**: ${p.id}\n**Mô tả**: ${p.plo}`).join('\n\n')}`

    const content = await askLLMOpenRouter(prompt, param)

    const analyzeBuffer = Buffer.from('# Kết quả phân tích PLO\n\n' + content, 'utf-8')

    // Trích xuất bảng Bloom từ content
    const bloomTable: any[] = []
    const lines = content.split('\n')
    let tableStarted = false
    for (const line of lines) {
      if (line.startsWith('| Mã PLO') || line.startsWith('|--------')) { tableStarted = true; continue }
      if (tableStarted && line.startsWith('|') && !line.startsWith('|--------')) {
        const row = line.split('|').slice(1, -1).map(x=>x.trim())
        if (row.length === 5) bloomTable.push(row)
      } else if (tableStarted && !line.startsWith('|')) break
    }

    const bloomWb = new ExcelJS.Workbook()
    const bloomWs = bloomWb.addWorksheet('Bloom Table')
    bloomWs.addRow(['Mã PLO', 'Mô tả rút gọn', 'Nhóm Năng lực', 'Tiến trình nhận thức', 'Loại kiến thức'])
    for (const row of bloomTable) bloomWs.addRow(row)
    const bloomBuffer = await bloomWb.xlsx.writeBuffer()

    return { analyzeBuffer, bloomBuffer: Buffer.from(bloomBuffer), bloomTable }
  } finally {
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile)
    }
  }
} 