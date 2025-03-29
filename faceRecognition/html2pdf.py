import base64

import pdfkit
import sys

def html_to_pdf(html_content, pdf_file):
    try:
        # 配置 wkhtmltopdf 的路径
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')  # Windows 路径

        # 将 HTML 字符串转换为 PDF
        pdf_bytes = pdfkit.from_string(html_content, False, configuration=config)

        with open(pdf_file, "wb") as f:
            f.write(pdf_bytes)

        print(f"PDF successfully created at {pdf_file}")
        return pdf_bytes
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # 从命令行参数获取 HTML 内容和 PDF 文件路径
    html_content = """
    <html>
      <body>
        <h1>我的简历</h1>
        <p>姓名：张三</p>
        <p>学历：本科</p>
        <p>工作经验：5年</p>
        <p>技能：Java, Python, React</p>
      </body>
    </html>
    """
    pdf_file = r"C:\workspace\AI-Interview\external\static\pdf\resume.pdf"
    pdf_bytes = html_to_pdf(html_content, pdf_file)

    if pdf_bytes:
        # 将二进制数据转换为 Base64 编码的字符串
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        print(base64_pdf)  # 输出 Base64 编码的 PDF