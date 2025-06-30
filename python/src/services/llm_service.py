from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzyARw95GU-P6HHfAKdAicnQNLWWQHC18"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import os
import logging
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentReportService:
    def __init__(self):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Dưới đây là thông tin học tập của một sinh viên:

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
- Phần "ĐÁNH GIÁ KỂT QUẢ HỌC TẬP" với bullet points
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

HTML phải hoàn chỉnh và render được trực tiếp trên browser.
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_html_report(self, student_data: str, output_dir: str = ".") -> tuple:
        try:
            logger.info("🔄 Đang tạo báo cáo HTML...")
            
            html_output = self.chain.run(text=student_data)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_filename = os.path.join(output_dir, f"bao_cao_hoc_tap_{timestamp}.html")
            
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            logger.info(f"✅ Đã tạo báo cáo HTML: {html_filename}")
            return html_filename, html_output
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo báo cáo HTML: {str(e)}")
            raise

    def convert_html_to_pdf(self, html_filename: str) -> str:
        """Chuyển đổi HTML sang PDF sử dụng ReportLab"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch, cm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from bs4 import BeautifulSoup
            
            pdf_filename = html_filename.replace('.html', '.pdf')
            logger.info(f"🔄 Đang chuyển đổi HTML sang PDF: {pdf_filename}")
            
            # Đăng ký font hỗ trợ tiếng Việt
            font_name = "Helvetica"  # Default font
            try:
                if platform.system() == "Windows":
                    # Sử dụng font có sẵn trên Windows
                    font_paths = [
                        "C:/Windows/Fonts/arial.ttf",
                        "C:/Windows/Fonts/ArialUnicodeMS.ttf", 
                        "C:/Windows/Fonts/calibri.ttf",
                        "C:/Windows/Fonts/times.ttf"
                    ]
                    
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            try:
                                pdfmetrics.registerFont(TTFont('VietnameseFont', font_path))
                                font_name = "VietnameseFont"
                                logger.info(f"✅ Đã đăng ký font: {font_path}")
                                break
                            except Exception as e:
                                logger.warning(f"⚠️ Không thể đăng ký font {font_path}: {e}")
                                continue
                                
            except Exception as e:
                logger.warning(f"⚠️ Sử dụng font mặc định do lỗi: {e}")
            
            # Đọc file HTML với encoding UTF-8
            with open(html_filename, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Tạo PDF document
            doc = SimpleDocTemplate(
                pdf_filename,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Tạo styles với font hỗ trợ tiếng Việt
            styles = getSampleStyleSheet()
            
            # Style cho tiêu đề chính
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontName=font_name,
                fontSize=18,
                spaceAfter=30,
                textColor=colors.HexColor('#2E86AB'),
                alignment=1,  # Center
            )
            
            # Style cho tiêu đề section
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=14,
                spaceAfter=12,
                textColor=colors.HexColor('#2E86AB'),
                leftIndent=0
            )
            
            # Style cho văn bản thường
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=11,
                spaceAfter=6,
                leading=14,
                textColor=colors.black
            )
            
            # Style cho thông tin sinh viên
            info_style = ParagraphStyle(
                'InfoStyle',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=10,
                textColor=colors.HexColor('#666666')
            )
            
            story = []
            
            # Lấy title
            title_tag = soup.find('h1')
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                story.append(Paragraph(title_text, title_style))
                story.append(Spacer(1, 20))
            
            # Lấy thông tin sinh viên từ table
            table_tag = soup.find('table')
            if table_tag:
                rows = table_tag.find_all('tr')
                table_data = []
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        table_data.append([key, value])
                
                if table_data:
                    # Tạo table cho thông tin sinh viên
                    t = Table(table_data, colWidths=[4*cm, 8*cm])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), font_name),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 20))
            
            # Lấy nội dung các section
            sections = soup.find_all(['h2', 'h3'])
            
            for section in sections:
                # Thêm tiêu đề section
                section_title = section.get_text(strip=True)
                if section_title and 'BÁO CÁO' not in section_title.upper():
                    story.append(Paragraph(section_title, heading_style))
                
                # Lấy nội dung sau section
                next_sibling = section.find_next_sibling()
                section_content = ""
                
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3']:
                    if next_sibling.name == 'p':
                        section_content += next_sibling.get_text(strip=True) + "<br/><br/>"
                    elif next_sibling.name == 'ul':
                        for li in next_sibling.find_all('li'):
                            section_content += "• " + li.get_text(strip=True) + "<br/>"
                        section_content += "<br/>"
                    elif next_sibling.name == 'div':
                        div_text = next_sibling.get_text(strip=True)
                        if div_text:
                            section_content += div_text + "<br/><br/>"
                    elif next_sibling.string:
                        section_content += next_sibling.string.strip() + " "
                    next_sibling = next_sibling.find_next_sibling()
                
                if section_content.strip():
                    # Làm sạch và format text
                    section_content = section_content.replace('<br/><br/><br/>', '<br/><br/>')
                    story.append(Paragraph(section_content, normal_style))
                    story.append(Spacer(1, 15))
            
            # Thêm footer
            footer_text = f"Báo cáo được tạo tự động vào {datetime.now().strftime('%d/%m/%Y lúc %H:%M:%S')}"
            story.append(Spacer(1, 30))
            story.append(Paragraph(footer_text, info_style))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ Đã tạo PDF thành công: {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo PDF: {e}")
            raise e

    def generate_complete_report(self, student_data: str, output_dir: str = ".") -> dict:
        """Tạo báo cáo hoàn chỉnh HTML + PDF"""
        try:
            # Tạo HTML report
            html_file, html_content = self.generate_html_report(student_data, output_dir)
            
            # Chuyển đổi sang PDF
            pdf_file = self.convert_html_to_pdf(html_file)
            
            return {
                'success': True,
                'html_file': html_file,
                'pdf_file': pdf_file,
                'html_content': html_content,
                'message': '✅ Báo cáo được tạo thành công!'
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo báo cáo: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f'❌ Có lỗi xảy ra: {str(e)}'
            }

# Instance để sử dụng
report_service = StudentReportService()

if __name__ == "__main__":
    # Test data
    sample_data = """
    Thông tin sinh viên:
    - Mã sinh viên: 20221234
    - Họ tên: Nguyễn Văn An
    - Lớp: CNTT01-K16
    - Số môn đã học: 20
    - Số học kỳ đã hoàn thành: 5
    - CPA hiện tại: 1.5
    - Tổng số tín chỉ tích lũy: 50
    - Cảnh báo học vụ hiện tại: Mức 2
    
    Chi tiết theo từng học kỳ:
    | Học kỳ | CPA   | Tín chỉ tích lũy | Cảnh báo học vụ |
    |--------|-------|------------------|------------------|
    | 20221  | 1.38  | 9.0              | Mức 0            |
    | 20222  | 2.18  | 14.0             | Mức 1            |
    | 20231  | 1.2   | 20.0             | Mức 2            |
    | 20232  | 2.0   | 35.0             | Mức 2            |
    | 20241  | 1.3   | 50.0             | Mức 1            |
    """
    
    print("🚀 Bắt đầu test tạo báo cáo...")
    result = report_service.generate_complete_report(sample_data)
    
    if result['success']:
        print(f"✅ {result['message']}")
        print(f"📄 HTML: {result['html_file']}")
        print(f"📄 PDF: {result['pdf_file']}")
    else:
        print(f"❌ {result['message']}")
