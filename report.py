"""PDF / CSV prediction reports."""
import csv
import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DISCLAIMER = (
    "This application is an academic machine learning project and is not a substitute "
    "for professional medical diagnosis. Predictions may be incorrect. Consult a "
    "qualified healthcare professional for medical advice."
)


def build_csv(rec: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Field", "Value"])
    rows = [
        ("Report generated", rec.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))),
        ("Patient name/ID", rec["patient_ref"]),
        ("Age", rec.get("age")),
        ("Weight (kg)", rec.get("weight_kg")),
        ("Height (cm)", rec.get("height_cm") or ""),
        ("BMI", rec.get("bmi") or ""),
        ("Symptoms", "; ".join(rec["symptoms"])),
        ("Predicted disease", rec["predicted_disease"]),
        ("Model used", rec["model_used"]),
    ]
    for name, pred in (rec.get("individual") or {}).items():
        rows.append((f"Prediction - {name}", pred))
    w.writerows(rows)
    w.writerow([])
    w.writerow(["Disclaimer", DISCLAIMER])
    return buf.getvalue().encode("utf-8")


def build_pdf(rec: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Disease Prediction Report")
    ss = getSampleStyleSheet()
    teal = colors.HexColor("#0E7C86")
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=teal, fontSize=20)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#12324A"))
    body = ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    def table(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E3F1F5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        return t

    p = lambda s: Paragraph(escape(str(s)), body)
    story = [
        Paragraph("Smart Disease Prediction System", h1),
        Paragraph("AI-Powered Disease Prediction Using Ensemble Learning", ss["Italic"]),
        Spacer(1, 12),
        Paragraph("Patient information", h2),
        table([
            [p("Patient name / ID"), p(rec["patient_ref"])],
            [p("Age"), p(rec.get("age", ""))],
            [p("Weight"), p(f"{rec.get('weight_kg', '')} kg")],
            [p("Height / BMI"), p(f"{rec.get('height_cm') or '-'} cm / {rec.get('bmi') or '-'}")],
            [p("Date"), p(rec.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M")))],
        ], [5 * cm, 12 * cm]),
        Spacer(1, 10),
        Paragraph("Reported symptoms", h2),
        p(", ".join(rec["symptoms"])),
        Spacer(1, 10),
        Paragraph("Prediction", h2),
        table([
            [p("Predicted disease"), Paragraph(f"<b>{escape(rec['predicted_disease'])}</b>", body)],
            [p("Model used"), p(rec["model_used"])],
        ], [5 * cm, 12 * cm]),
    ]
    if rec.get("individual"):
        story += [Spacer(1, 10), Paragraph("Individual model predictions", h2),
                  table([[p(k), p(v)] for k, v in rec["individual"].items()], [5 * cm, 12 * cm])]
    if rec.get("scores"):
        story += [Spacer(1, 10), Paragraph("Model confidence scores (top 5)", h2),
                  table([[p(c), p(f"{s}%")] for c, s in rec["scores"]], [5 * cm, 12 * cm])]
    story += [Spacer(1, 18), Paragraph("<b>Disclaimer:</b> " + escape(DISCLAIMER), small)]
    doc.build(story)
    return buf.getvalue()
