import * as ExcelJS from 'exceljs'
import axios from 'axios'
import * as fs from 'fs'
import * as path from 'path'
import { v4 as uuidv4 } from 'uuid'
import { ConfigService } from '../../config/config.service'

const DEFAULT_CLO_SUGGEST_PROMPT = `Bạn là một chuyên gia thiết kế chương trình đào tạo đại học.
Cho đề cương học phần sau, hãy đề xuất **{num_clo} Chuẩn đầu ra học phần (CLO)**.
Yêu cầu: mỗi CLO cụ thể, đo lường được, có động từ Bloom, phủ kiến thức/kỹ năng/thái độ, trình bày gạch đầu dòng.
Ngôn ngữ: tiếng Việt.

Phản hồi chỉ gồm **Danh sách CLO**; không giải thích thêm.`

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
): Promise<{ markdownBuffer: Buffer, excelBuffer: Buffer, cloList: string[] }> {
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

    const cloLines = extractCLOLines(content)
    
    const workbook = new ExcelJS.Workbook()
    const worksheet = workbook.addWorksheet('CLO Suggestions')
    
    worksheet.addRow(['STT', 'CLO đề xuất'])
    cloLines.forEach((clo, index) => {
      worksheet.addRow([index + 1, clo])
    })

    worksheet.getColumn(1).width = 10
    worksheet.getColumn(2).width = 80
    worksheet.getRow(1).font = { bold: true }

    const excelBuffer = await workbook.xlsx.writeBuffer()

    return { 
      markdownBuffer, 
      excelBuffer: Buffer.from(excelBuffer), 
      cloList: cloLines 
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