import * as ExcelJS from 'exceljs'
import axios from 'axios'
import * as fs from 'fs'
import * as path from 'path'
import { v4 as uuidv4 } from 'uuid'
import { ConfigService } from '../../config/config.service'

const DEFAULT_CLO_SUGGEST_PROMPT = `Bạn là một chuyên gia thiết kế chương trình đào tạo đại học.
Cho đề cương học phần sau, hãy đề xuất **{num_clo} Chuẩn đầu ra học phần (CLO)**.
Yêu cầu: mỗi CLO cụ thể, đo lường được, có động từ Bloom, phủ kiến thức/kỹ năng/thái độ.
Ngôn ngữ: tiếng Việt.

## Ràng buộc chất lượng
- **Mỗi CLO** là **một câu** mô tả hành vi có thể **đánh giá** được, dùng **động từ Bloom** rõ ràng.  (Heading 2)
- Gắn nhãn **I/R/M** ngay cuối câu để thể hiện mức độ curriculum mapping.
  *I* = Introduce, *R* = Reinforce, *M* = Master.
- **Giải thích lý do** lựa chọn mức I/R/M: nêu rõ *phần, chương hoặc hoạt động* trong syllabus hỗ trợ người học đạt mức đó.  (Heading 3)
- **Cân bằng cấp độ tư duy Bloom** – ít nhất một CLO ở mức *Phân tích/Đánh giá/Sáng tạo*.
- **Không thêm nội dung ngoài syllabus**; nếu thiếu thông tin, ghi “(chưa đủ dữ liệu)” ở phần giải thích.

**BẮT BUỘC**: Trả về kết quả dưới dạng bảng Markdown với format chính xác:

| STT | Mã CLO | Nội dung | Mức độ tư duy | Giải thích |
|-----|--------|----------|---------------|-----------|
| 1   | CLO1   | Mô tả chi tiết CLO1 | I | Cơ sở từ chương/phần nào trong syllabus |
| 2   | CLO2   | Mô tả chi tiết CLO2 | R | Cơ sở từ chương/phần nào trong syllabus |
| 3   | CLO3   | Mô tả chi tiết CLO3 | M | Cơ sở từ chương/phần nào trong syllabus |

Lưu ý mức độ tư duy: 
- I (Introduce): Giới thiệu kiến thức cơ bản
- R (Reinforce): Củng cố và vận dụng
- M (Master): Thành thạo và sáng tạo

Cột "Giải thích" phải nêu rõ cơ sở từ syllabus (chương, phần, nội dung cụ thể) làm căn cứ cho CLO đó.

Chỉ trả về bảng markdown, không có text hay giải thích nào khác.`

const DEFAULT_CLO_CHECK_PROMPT = `Bạn là chuyên gia đánh giá chuẩn đầu ra học phần (CLO) theo OBE/Bloom.
Dựa trên đề cương học phần và danh sách CLO hiện có, hãy lập bảng nhận xét gồm 4 cột:
| # | Nội dung CLO | I/R/M | Nhận xét/Justification |
Yêu cầu:
- Xác định mức I/R/M thích hợp dựa trên phạm vi và độ sâu kiến thức trong syllabus.
- Nhận xét ngắn gọn (≤40 từ) về sự đầy đủ, động từ Bloom, và mức alignment.
- Nếu CLO mơ hồ hoặc thiếu căn cứ, đánh dấu ⚠️ trong cột Nhận xét.
- Không thêm CLO mới. Trả về bảng Markdown duy nhất, không giải thích ngoài bảng.`

function getParam(paramBuffer?: Buffer) {
  if (!paramBuffer) return {}
  try {
    const param = JSON.parse(paramBuffer.toString())
    if (param && typeof param === 'object') return param
    return {}
  } catch {
    return {}
  }
}

async function askLLMOpenRouter(prompt: string, param: any = {}, configService: ConfigService) {
  try {
    console.log('Request params:', JSON.stringify(param, null, 2))
    
    const config = configService?.openRouterConfig
    const api_key = param?.api_key || param?.apiKey || config?.apiKey
    const model = param?.model || config?.model
    const temperature = param?.temperature ?? config?.temperature
    const system_instruction = param?.system_instruction || param?.systemInstruction || config?.systemInstruction
    const max_tokens = param?.max_tokens || param?.maxTokens || 2048
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
      max_tokens,
      temperature
    }
    
    const response = await axios.post(url, payload, { headers })
    const result = response.data
    if (result.error) {
      console.log(result.error)
      throw new Error(result.error)
    }
    return result.choices[0].message.content || ''
  } catch (error) {
    console.error('Error in askLLMOpenRouter:', error)
    throw error
  }
}

function extractCLOTable(markdownText: string): { stt: number, maCLO: string, noiDung: string, mucDoTuDuy: string, giaiThich: string }[] {
  const lines = markdownText.split('\n')
  const cloTable: { stt: number, maCLO: string, noiDung: string, mucDoTuDuy: string, giaiThich: string }[] = []
  let tableStarted = false
  
  for (const line of lines) {
    if (line.includes('| STT') || line.includes('|--')) {
      tableStarted = true
      continue
    }
    
    if (tableStarted && line.startsWith('|') && !line.includes('--')) {
      const row = line.split('|').slice(1, -1).map(x => x.trim())
      if (row.length >= 4) {
        const stt = parseInt(row[0]) || 0
        const maCLO = row[1] || ''
        const noiDung = row[2] || ''
        const mucDoTuDuy = row[3] || ''
        const giaiThich = row[4] || ''
        
        if (stt > 0 && maCLO && noiDung) {
          cloTable.push({ stt, maCLO, noiDung, mucDoTuDuy, giaiThich })
        }
      }
    } else if (tableStarted && !line.startsWith('|') && line.trim() !== '') {
      break
    }
  }
  
  return cloTable
}

function extractCLOLines(markdownText: string): string[] {
  return markdownText
    .split('\n')
    .filter(line => line.trim().match(/^[-*•]\s+/))
    .map(line => line.replace(/^[-*•]\s+/, '').trim())
    .filter(line => line.length > 0)
}

export async function suggestCLOFromSyllabus(
  syllabusBuffer: Buffer,
  paramBuffer?: Buffer,
  bodyPrompt?: string,
  configService?: ConfigService
): Promise<{ markdownBuffer: Buffer, excelBuffer: Buffer, cloTable: { stt: number, maCLO: string, noiDung: string, mucDoTuDuy: string, giaiThich: string }[] }> {
  try {
    if (!syllabusBuffer || syllabusBuffer.length === 0) {
      throw new Error('Syllabus buffer is empty or invalid')
    }

    const syllabusText = syllabusBuffer.toString('utf-8')
    const param = getParam(paramBuffer)
    const customPrompt = bodyPrompt || param?.prompt || DEFAULT_CLO_SUGGEST_PROMPT
    const numCLO = param?.numCLO || param?.num_clo || 4
    
    const prompt = customPrompt.replace('{num_clo}', numCLO) + 
      `\n\n# Đề cương học phần\n${syllabusText}`

    const content = await askLLMOpenRouter(prompt, param, configService)
    
    const markdownContent = `# Đề xuất CLO cho học phần\n\n## Ngày tạo: ${new Date().toLocaleString('vi-VN')}\n\n## Số CLO được đề xuất: ${numCLO}\n\n## Kết quả đề xuất:\n\n${content}`
    const markdownBuffer = Buffer.from(markdownContent, 'utf-8')

    let cloTable = extractCLOTable(content)
    
    // Fallback to bullet points if table parsing fails
    if (cloTable.length === 0) {
      const cloLines = extractCLOLines(content)
      cloTable = cloLines.map((line, index) => ({
        stt: index + 1,
        maCLO: `CLO${index + 1}`,
        noiDung: line,
        mucDoTuDuy: 'N/A',
        giaiThich: 'N/A'
      }))
    }
    
    const workbook = new ExcelJS.Workbook()
    const worksheet = workbook.addWorksheet('CLO Suggestions')
    
    worksheet.addRow(['STT', 'Mã CLO', 'Nội dung', 'Mức độ tư duy', 'Giải thích'])
    cloTable.forEach((clo) => {
      worksheet.addRow([clo.stt, clo.maCLO, clo.noiDung, clo.mucDoTuDuy, clo.giaiThich])
    })

    worksheet.getColumn(1).width = 10
    worksheet.getColumn(2).width = 15
    worksheet.getColumn(3).width = 80
    worksheet.getColumn(4).width = 15
    worksheet.getColumn(5).width = 50
    worksheet.getRow(1).font = { bold: true }

    const excelBuffer = await workbook.xlsx.writeBuffer()

    return { 
      markdownBuffer, 
      excelBuffer: Buffer.from(excelBuffer), 
      cloTable: cloTable 
    }
  } catch (error) {
    console.error('Error in suggestCLOFromSyllabus:', error)
    throw error
  }
}

export async function checkCLOAgainstSyllabus(
  syllabusBuffer: Buffer,
  cloBuffer: Buffer,
  paramBuffer?: Buffer,
  bodyPrompt?: string,
  configService?: ConfigService
): Promise<{ markdownBuffer: Buffer, excelBuffer: Buffer, evaluationTable: any[] }> {
  try {
    if (!syllabusBuffer || syllabusBuffer.length === 0) {
      throw new Error('Syllabus buffer is empty or invalid')
    }
    if (!cloBuffer || cloBuffer.length === 0) {
      throw new Error('CLO buffer is empty or invalid')
    }

    const syllabusText = syllabusBuffer.toString('utf-8')
    const cloText = cloBuffer.toString('utf-8')
    const param = getParam(paramBuffer)
    const customPrompt = bodyPrompt || param?.prompt || DEFAULT_CLO_CHECK_PROMPT
    
    const prompt = `${customPrompt}\n\n## Đề cương\n${syllabusText}\n\n## Danh sách CLO hiện tại\n${cloText}`

    const content = await askLLMOpenRouter(prompt, param, configService)
    
    const markdownContent = `# Đánh giá CLO theo đề cương học phần\n\n## Ngày đánh giá: ${new Date().toLocaleString('vi-VN')}\n\n## Kết quả đánh giá:\n\n${content}`
    const markdownBuffer = Buffer.from(markdownContent, 'utf-8')

    const evaluationTable: any[] = []
    const lines = content.split('\n')
    let tableStarted = false
    
    for (const line of lines) {
      if (line.includes('| #') || line.includes('|--')) {
        tableStarted = true
        continue
      }
      if (tableStarted && line.startsWith('|') && !line.includes('--')) {
        const row = line.split('|').slice(1, -1).map(x => x.trim())
        if (row.length >= 4) {
          evaluationTable.push({
            'STT': row[0],
            'Nội dung CLO': row[1],
            'I/R/M': row[2],
            'Nhận xét/Justification': row[3]
          })
        }
      } else if (tableStarted && !line.startsWith('|')) {
        break
      }
    }

    const workbook = new ExcelJS.Workbook()
    const worksheet = workbook.addWorksheet('CLO Evaluation')
    
    if (evaluationTable.length > 0) {
      const headers = Object.keys(evaluationTable[0])
      worksheet.addRow(headers)
      
      evaluationTable.forEach(row => {
        worksheet.addRow(Object.values(row))
      })

      worksheet.getColumn(1).width = 10
      worksheet.getColumn(2).width = 60
      worksheet.getColumn(3).width = 15
      worksheet.getColumn(4).width = 50
      worksheet.getRow(1).font = { bold: true }
    } else {
      worksheet.addRow(['STT', 'Nội dung CLO', 'I/R/M', 'Nhận xét/Justification'])
      worksheet.addRow(['Không tìm thấy bảng đánh giá trong phản hồi của LLM'])
    }

    const excelBuffer = await workbook.xlsx.writeBuffer()

    return { 
      markdownBuffer, 
      excelBuffer: Buffer.from(excelBuffer), 
      evaluationTable 
    }
  } catch (error) {
    console.error('Error in checkCLOAgainstSyllabus:', error)
    throw error
  }
} 