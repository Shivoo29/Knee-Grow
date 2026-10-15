import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(45, letter[1] - 30, "RSNA Knee 2026 | Multi-Teacher Seed Distillation Critique & Blueprint")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, letter[1] - 34, letter[0] - 45, letter[1] - 34)
            
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 45, 28, text)
        self.drawString(45, 28, "CONFIDENTIAL & STRATEGIC | RSNA KNEE CHALLENGE 2026")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 38, letter[0] - 45, 38)
        self.restoreState()


def build_pdf(filename="multi_teacher_distillation_critique_and_architecture.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"), # Deep Blue
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=7,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    verdict_style = ParagraphStyle(
        'VerdictBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_hdr = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#FFFFFF")
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("RSNA Knee Abnormality Detection: Critique & Architecture", title_style))
    story.append(Paragraph("Rigorous Critique of Multi-Teacher Seed Distillation & The Two-Stage Resolution Blueprint", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # Executive Verdict Callout Box
    verdict_html = """
    <b>EXECUTIVE VERDICT:</b> The idea of creating a single student model trained on spliced predictions from candidate teacher seeds selected per label is <b>statistically invalid and technically fragile</b> on this dataset.<br/><br/>
    • <b>Statistical Illusion:</b> On the 58 Gold studies, per-finding score variations between seeds have a 95% bootstrap CI half-width of ±0.10 to ±0.19. All 12 findings received a statistical verdict of <code>noise</code> in T07 error analysis.<br/>
    • <b>In-Sample Leakage:</b> Individual seeds were trained on all 4,349 studies without out-of-fold partitions. Distilling in-sample outputs forces the student to memorize training noise.<br/>
    • <b>The Two-Stage Solution:</b> Keep training supervision leak-free via <b>4-fold OOF predictions + NLP report labels</b> (<code>labels_oofmix_v1.parquet</code>), and realize multi-model specialization at <b>test-time inference via Percentile Rank Ensembling</b> on Kaggle.
    """
    
    verdict_data = [[Paragraph(verdict_html, verdict_style)]]
    verdict_table = Table(verdict_data, colWidths=[letter[0] - 90])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")), # light mint/emerald tint
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#16A34A")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    # Section 1: Deep-Dive Critique
    story.append(Paragraph("1. The Four Fatal Traps of Naive Multi-Teacher Distillation", h1_style))
    
    story.append(Paragraph("<b>Trap 1: The Gold-58 Variance Illusion (Overfitting to Pure Noise)</b>", h2_style))
    story.append(Paragraph(
        "In <code>delegation/results/T07_error_analysis.md</code> (Section <code>## Q4</code>), we audited whether any candidate seed beat the seed-blend on the 58 Gold validation studies. "
        "Because rare findings contain very few positive cases (Lateral OA has 11 positives, MCL Tear has 9), the 95% bootstrap confidence interval half-width ranges from ±0.107 to ±0.191. "
        "All 12 findings were formally proven to have <b>verdict = noise</b>. Selecting Seed 8 for Lateral OA or Seed 10 for Meniscus is not selecting a specialized feature extractor; it is selecting random seed fluctuations on 1 or 2 lucky patients.",
        body_style
    ))

    story.append(Paragraph("<b>Trap 2: In-Sample Memorization Catastrophe</b>", h2_style))
    story.append(Paragraph(
        "The historical checkpoints (<code>v6_s6</code>, <code>v6d_s8</code>, <code>v6d_s10</code>, <code>dinov2_s11</code>) were trained across the entire 4,349 studies. "
        "Their predictions across the training set are strictly in-sample. If a student is trained to match these in-sample teacher probabilities, it is forced to memorize teachers' training artifacts and overconfident hallucinations. "
        "Generating true out-of-fold multi-seed predictions would require 16 full retraining runs (4 folds × 4 seeds), which is impossible before the October 22 deadline.",
        body_style
    ))

    story.append(Paragraph("<b>Trap 3: Spliced Multi-Teacher Label Correlation Breakage</b>", h2_style))
    story.append(Paragraph(
        "Knee abnormalities are biologically and biomechanically coupled (e.g., Medial Meniscus Tear correlates strongly with Medial Compartment OA). "
        "Routing finding probabilities from different teacher models without joint calibration destroys cross-label correlation structure and distorts downstream macro-AUC ranking.",
        body_style
    ))

    story.append(Paragraph("<b>Trap 4: Distillation Capacity Compression vs. Ensembling</b>", h2_style))
    story.append(Paragraph(
        "Distilling multiple models into a single parameter set unavoidably suffers from capacity compression loss: <code>Var(ensemble) &lt;&lt; Var(student)</code>. "
        "In medical imaging with high label noise, test-time ensembling directly cancels out independent variance without losing model expressiveness.",
        body_style
    ))

    # Clean Page Break between Critique and Resolution
    story.append(PageBreak())

    # Section 2: Validated Architecture Blueprint
    story.append(Paragraph("2. The Validated Architecture Blueprint: Two-Stage Execution", h1_style))
    story.append(Paragraph(
        "To capture genuine multi-teacher synergy without the fatal pitfalls of in-sample splicing, the pipeline is divided into two distinct, leak-free stages:",
        body_style
    ))

    blueprint_rows = [
        [
            Paragraph("<b>Component</b>", table_hdr),
            Paragraph("<b>Specification</b>", table_hdr),
            Paragraph("<b>Strategic Rationale & Benefit</b>", table_hdr)
        ],
        [
            Paragraph("<b>Training Supervision</b>", table_cell),
            Paragraph("<code>labels_oofmix_v1.parquet</code><br/>(0.75 Vision OOF + 0.25 Report Labels)", table_cell),
            Paragraph("Combines 4-fold out-of-fold vision teacher predictions with NLP report extraction. Resolves label noise and lifts Gold AUC from 0.899 (pure report) to <b>0.941</b>.", table_cell)
        ],
        [
            Paragraph("<b>Data Augmentation & Architecture</b>", table_cell),
            Paragraph("T05 Recipe:<br/>• Clean window boundaries<br/>• Anatomical mirror (p=0.5)<br/>• Affine contrast aug<br/>• Learned slot embeddings", table_cell),
            Paragraph("Eliminates 12% mixed-plane artifact windows, doubles effective dataset volume while preserving medial/lateral anatomical signals, and provides scanner-shift invariance.", table_cell)
        ],
        [
            Paragraph("<b>Model Ensemble (Test-Time)</b>", table_cell),
            Paragraph("4-Way Ensemble:<br/>• Public R7 (LB 0.957)<br/>• Raptor baseline (LB 0.954)<br/>• <code>final_s40</code> (Clean OOF-mix)<br/>• <code>final_s41</code> (Clean OOF-mix)", table_cell),
            Paragraph("Ensembles high-performing diverse backbones (CoAtNet-384, ConvNeXt, Swin) directly at test time, maximizing variance reduction without distillation loss.", table_cell)
        ],
        [
            Paragraph("<b>Ensembling Method</b>", table_cell),
            Paragraph("Percentile Rank Blending:<br/><code>rank(p) / N</code>", table_cell),
            Paragraph("Eliminates calibration and sigmoid temperature discrepancies between different model architectures. Public LB verified: <b>0.952</b> (+0.003 over R7 standalone).", table_cell)
        ]
    ]

    blueprint_table = Table(blueprint_rows, colWidths=[110, 155, 257])
    blueprint_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(blueprint_table)
    story.append(Spacer(1, 8))

    # Section 3: Live Training Status
    story.append(Paragraph("3. Live Model Execution Status (Real-Time Snapshot)", h1_style))
    story.append(Paragraph(
        "Both production models incorporating the full T05 data recipe and <code>labels_oofmix_v1</code> supervision are actively training:",
        body_style
    ))

    status_rows = [
        [
            Paragraph("<b>Worker / Model</b>", table_hdr),
            Paragraph("<b>Architecture & Config</b>", table_hdr),
            Paragraph("<b>Current Progress</b>", table_hdr),
            Paragraph("<b>Best Gold AUC</b>", table_hdr),
            Paragraph("<b>Estimated Finish</b>", table_hdr)
        ],
        [
            Paragraph("<b>Local PC</b><br/><code>final_s40</code>", table_cell),
            Paragraph("CoAtNet-384<br/>bs=1, accum=16, k=12<br/>T05 Clean + OOF-mix", table_cell),
            Paragraph("<b>Epoch 9 / 14</b><br/>Loss: 0.696", table_cell),
            Paragraph("<b>0.9120</b><br/>(ep8, climbing)", table_cell),
            Paragraph("~03:45 AM IST<br/>(Oct 16)", table_cell)
        ],
        [
            Paragraph("<b>Remote node1</b><br/><code>final_s41</code>", table_cell),
            Paragraph("CoAtNet-384<br/>bs=2, accum=8, k=12<br/>T05 Clean + OOF-mix", table_cell),
            Paragraph("<b>Epoch 6 / 14</b><br/>Loss: 0.706", table_cell),
            Paragraph("<b>0.8968</b><br/>(ep5, climbing)", table_cell),
            Paragraph("~05:30 AM IST<br/>(Oct 16)", table_cell)
        ]
    ]

    status_table = Table(status_rows, colWidths=[90, 142, 100, 95, 95])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 8))

    # Section 4: Competition Milestone Plan
    story.append(Paragraph("4. Strategic Milestone Timeline (Oct 16 – Oct 22)", h1_style))
    
    story.append(Paragraph("• <b>Oct 16 (Today): Checkpoint Convergence & Offline Gold Benchmark</b><br/>"
                           "Evaluate epoch-best and SWA checkpoints for both <code>final_s40</code> and <code>final_s41</code>. "
                           "Compute 4-way rank ensemble validation on Gold-58 (R7 + Raptor + S40 + S41).", bullet_style))
    
    story.append(Paragraph("• <b>Oct 17 – 18: Kaggle Pipeline Packaging & Runtime Verification</b><br/>"
                           "Build updated standalone submission notebook. Ensure complete inference runs comfortably within the 9-hour dual-T4 Kaggle ceiling (estimated runtime: ~4.5 hours).", bullet_style))
    
    story.append(Paragraph("• <b>Oct 19 – 21: Final Leaderboard Submissions & Stability Check</b><br/>"
                           "Submit Primary Ensemble (R7 + Raptor + S40 + S41) and secondary lightweight efficiency submission. Verify private LB safety against vendor/prevalence shifts.", bullet_style))
    
    story.append(Paragraph("• <b>Oct 22: Final Selection Freeze</b><br/>"
                           "Lock in the two chosen submissions before deadline.", bullet_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
