"""
report_pdf.py — AI Medical Diagnosis PDF Generator
===================================================
Creates a professional, styled PDF medical report using ReportLab.
Called by main.py /api/report/pdf endpoint.

Usage:
    pdf_path = create_pdf_report(
        disease="PNEUMONIA",
        confidence=93.42,
        severity="Moderate",
        findings="Lung opacity indicating pneumonia infection.",
        recommendation="Immediate medical consultation required.",
        image=pil_image,
    )
"""

import os
import tempfile
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, HRFlowable,
)


# ── colour palette ─────────────────────────────────────────────────────────────
_BLUE      = colors.HexColor("#2563eb")
_DARK      = colors.HexColor("#0f172a")
_LIGHT_BG  = colors.HexColor("#f0f4ff")
_ALT_ROW   = colors.HexColor("#ffffff")
_MUTED     = colors.HexColor("#6b7a99")
_GREEN     = colors.HexColor("#00c853")
_AMBER     = colors.HexColor("#ff9800")
_RED       = colors.HexColor("#f44336")

_DISEASE_COLOR = {
    "NORMAL":       _GREEN,
    "PNEUMONIA":    _AMBER,
    "TUBERCULOSIS": _RED,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _style(name, **kwargs):
    base = getSampleStyleSheet()["Normal"]
    return ParagraphStyle(name, parent=base, **kwargs)


def _section(title: str, style) -> list:
    return [Paragraph(title, style), Spacer(1, 4)]


# ── public API ─────────────────────────────────────────────────────────────────

def create_pdf_report(
    disease: str,
    confidence: float,
    severity: str,
    findings: str,
    recommendation: str,
    image: Image.Image,
) -> str:
    """
    Generate a styled A4 PDF report.

    Parameters
    ----------
    disease        : "NORMAL" | "PNEUMONIA" | "TUBERCULOSIS"
    confidence     : 0–100
    severity       : "None" | "Moderate" | "High"
    findings       : radiological finding text
    recommendation : clinical recommendation text
    image          : PIL Image of the uploaded X-ray

    Returns
    -------
    str — absolute path to the generated temporary PDF file.
          Caller is responsible for deleting it after streaming.
    """
    disease = disease.upper()

    # ── temp files ────────────────────────────────────────────────────────────
    tmp        = tempfile.gettempdir()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path   = os.path.join(tmp, f"medical_report_{timestamp}.pdf")
    img_path   = os.path.join(tmp, f"xray_{timestamp}.jpg")

    # Save X-ray thumbnail (square, 400 px)
    thumb = image.convert("RGB")
    thumb.thumbnail((400, 400))
    thumb.save(img_path, "JPEG", quality=90)

    # ── document ──────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
    )

    # ── styles ────────────────────────────────────────────────────────────────
    s_title = _style(
        "ReportTitle",
        fontSize=22, spaceAfter=4,
        textColor=_DARK, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    s_sub = _style(
        "ReportSub",
        fontSize=10, spaceAfter=2,
        textColor=_MUTED, alignment=TA_CENTER,
    )
    s_section = _style(
        "Section",
        fontSize=13, spaceBefore=14, spaceAfter=6,
        textColor=_BLUE, fontName="Helvetica-Bold",
    )
    s_body = _style(
        "Body",
        fontSize=11, spaceAfter=4,
        textColor=colors.HexColor("#333333"), leading=17,
    )
    s_disclaimer = _style(
        "Disclaimer",
        fontSize=8.5, textColor=_MUTED, alignment=TA_CENTER, leading=13,
    )

    # ── story ─────────────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("🩺  AI MEDICAL DIAGNOSIS REPORT", s_title))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  ·  %H:%M UTC')}",
        s_sub,
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=_BLUE, spaceAfter=12))

    # ── Diagnosis summary table ───────────────────────────────────────────────
    risk = "Low" if disease == "NORMAL" else "High"
    d_color = _DISEASE_COLOR.get(disease, _MUTED)

    rows = [
        ["Field", "Result"],
        ["Disease Detected", disease],
        ["Confidence Level", f"{confidence:.2f}%"],
        ["Severity", severity],
        ["Risk Level", risk],
        ["Report Date", datetime.now().strftime("%d %B %Y")],
    ]

    tbl = Table(rows, colWidths=[75 * mm, 95 * mm])
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0), _BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        # Data rows
        ("BACKGROUND",  (0, 1), (-1, 1), _LIGHT_BG),
        ("BACKGROUND",  (0, 2), (-1, 2), _ALT_ROW),
        ("BACKGROUND",  (0, 3), (-1, 3), _LIGHT_BG),
        ("BACKGROUND",  (0, 4), (-1, 4), _ALT_ROW),
        ("BACKGROUND",  (0, 5), (-1, 5), _LIGHT_BG),
        # Disease cell gets disease colour
        ("TEXTCOLOR",   (1, 1), (1, 1), d_color),
        ("FONTNAME",    (1, 1), (1, 1), "Helvetica-Bold"),
        # General
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING",  (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7f0")),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # ── X-ray image ───────────────────────────────────────────────────────────
    *_section("X-ray Image", s_section), [story.append(x) for x in _section("X-ray Image", s_section)]
    story.append(RLImage(img_path, width=110 * mm, height=110 * mm, kind="proportional"))
    story.append(Spacer(1, 14))

    # ── Clinical findings ─────────────────────────────────────────────────────
    for x in _section("Clinical Findings", s_section):
        story.append(x)
    story.append(Paragraph(findings, s_body))
    story.append(Spacer(1, 10))

    # ── Recommendations ───────────────────────────────────────────────────────
    for x in _section("Recommendations", s_section):
        story.append(x)
    story.append(Paragraph(recommendation, s_body))
    story.append(Spacer(1, 16))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=8))
    story.append(Paragraph(
        "⚠  DISCLAIMER: This report is AI-generated for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified and licensed healthcare professional before making "
        "any medical decisions.",
        s_disclaimer,
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)

    # clean up temp image
    try:
        os.remove(img_path)
    except OSError:
        pass

    return pdf_path