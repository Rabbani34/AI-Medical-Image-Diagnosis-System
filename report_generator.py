from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(filename, disease, confidence):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("<b>Medical Image Analysis Report</b>", styles["Title"]))
    content.append(Paragraph(f"Disease Detected: {disease}", styles["Normal"]))
    content.append(Paragraph(f"Confidence Score: {confidence:.2f}%", styles["Normal"]))
    content.append(Paragraph("This report is generated using AI-based medical image analysis system.", styles["Italic"]))

    doc.build(content)
