"""
build_report.py
Assembles the Week 6 PDF report: Week6_Aarav.pdf
Registers DejaVu Sans for full Unicode coverage (sigma, rho, etc.)
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVuSans", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", f"{FONT_DIR}/DejaVuSans-Oblique.ttf"))

PLOTS = "/home/claude/week6/plots"
OUT_PATH = "/mnt/user-data/outputs/Week6_Aarav.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleDV", fontName="DejaVuSans-Bold", fontSize=20,
                           leading=24, spaceAfter=6, textColor=colors.HexColor("#0f172a")))
styles.add(ParagraphStyle(name="SubtitleDV", fontName="DejaVuSans", fontSize=11,
                           leading=15, spaceAfter=16, textColor=colors.HexColor("#475569")))
styles.add(ParagraphStyle(name="H1DV", fontName="DejaVuSans-Bold", fontSize=14,
                           leading=18, spaceBefore=14, spaceAfter=8,
                           textColor=colors.HexColor("#1e3a8a")))
styles.add(ParagraphStyle(name="H2DV", fontName="DejaVuSans-Bold", fontSize=11.5,
                           leading=15, spaceBefore=8, spaceAfter=6,
                           textColor=colors.HexColor("#1e3a8a")))
styles.add(ParagraphStyle(name="BodyDV", fontName="DejaVuSans", fontSize=10,
                           leading=14.5, spaceAfter=6, textColor=colors.HexColor("#1f2937")))
styles.add(ParagraphStyle(name="BulletDV", fontName="DejaVuSans", fontSize=10,
                           leading=14, spaceAfter=3, textColor=colors.HexColor("#1f2937")))
styles.add(ParagraphStyle(name="CaptionDV", fontName="DejaVuSans-Oblique", fontSize=9,
                           leading=12, spaceAfter=12, textColor=colors.HexColor("#64748b")))
styles.add(ParagraphStyle(name="MonoDV", fontName="DejaVuSans", fontSize=9,
                           leading=13, spaceAfter=6, textColor=colors.HexColor("#1f2937"),
                           backColor=colors.HexColor("#f1f5f9"), borderPadding=6))

story = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
story.append(Paragraph("Week 6 — Neural Network Pricer for American Puts", styles["TitleDV"]))
story.append(Paragraph(
    "Aarav &nbsp;|&nbsp; Options Pricing Course &nbsp;|&nbsp; Training an MLP to match "
    "CRR binomial American put prices", styles["SubtitleDV"]))

# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
story.append(Paragraph("Part A — Dataset Generation", styles["H1DV"]))
story.append(Paragraph(
    "12,000 synthetic American put contracts were sampled uniformly over the ranges below "
    "and labeled with the Week 4 CRR binomial pricer (<b>crr_put_price</b>) at a "
    "<b>fixed 200-step</b> tree, so every label uses the same discretization error.",
    styles["BodyDV"]))

range_table_data = [
    ["Feature", "Range", "Rationale"],
    ["S0 (spot)", "U[50, 150]", "generic equity price scale"],
    ["K (strike)", "U[50, 150]", "same scale as S0 -> moneyness spans ~0.33x-3.0x"],
    ["T (maturity, yrs)", "U[0.05, 2.0]", "2 weeks to 2 years"],
    ["r (risk-free rate)", "U[0.00, 0.10]", "0%-10% annual, continuously compounded"],
    ["sigma (volatility)", "U[0.05, 0.60]", "5% low-vol blue chip to 60% high-vol growth stock"],
]
t = Table(range_table_data, colWidths=[3.4*cm, 3.0*cm, 8.6*cm])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "DejaVuSans"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 8))

story.append(Paragraph("<b>Label sanity checks (all passed on 12,000/12,000 rows):</b>", styles["BodyDV"]))
story.append(ListFlowable([
    ListItem(Paragraph("Finite: no NaN/Inf prices.", styles["BulletDV"])),
    ListItem(Paragraph("Non-negative: no price below 0.", styles["BulletDV"])),
    ListItem(Paragraph("At least intrinsic value: no price below max(K - S0, 0).", styles["BulletDV"])),
], bulletType="bullet", start="circle"))
story.append(Paragraph(
    "The labeled dataset is saved to <b>data/american_put_dataset.csv</b> so training "
    "can be rerun without regenerating labels (label generation takes ~40s for 12,000 "
    "contracts at 200 steps).", styles["BodyDV"]))

# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
story.append(Paragraph("Part B — Training", styles["H1DV"]))
story.append(Paragraph(
    "The dataset was split 80/10/10 into train/validation/test with a fixed seed (42). "
    "Features were standardized (z-score) using <b>training-set mean/std only</b>, then "
    "applied unchanged to validation and test to avoid leakage.", styles["BodyDV"]))
story.append(Paragraph(
    "<b>Architecture:</b> a PyTorch MLP, 5 -> 64 -> 64 -> 32 -> 1 (three hidden layers, "
    "ReLU activations), trained with Adam (lr=1e-3), MSE loss, batch size 128, for up to "
    "200 epochs with early stopping (patience 25 epochs on validation MSE). The checkpoint "
    "with the <b>lowest validation MSE</b> was saved and used for all downstream evaluation "
    "&mdash; not simply the final epoch.", styles["BodyDV"]))

story.append(Image(f"{PLOTS}/learning_curve.png", width=13*cm, height=8.4*cm))
story.append(Paragraph(
    "Figure 1. Train/validation MSE vs epoch (log scale). Both curves fall smoothly and "
    "track each other closely, with no sign of overfitting divergence; the best "
    "validation MSE was 0.0212 (val RMSE &asymp; 0.15), reached at the final epoch (200) "
    "before early stopping would have triggered.", styles["CaptionDV"]))

# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
story.append(Paragraph("Part C — Evaluation and Finance Checks", styles["H1DV"]))
story.append(Paragraph("Test-set error metrics (1,200 held-out contracts, never seen in training or model selection):",
                        styles["BodyDV"]))

metrics_table = [
    ["Metric", "Value"],
    ["MAE", "0.0883"],
    ["RMSE", "0.1230"],
    ["Max absolute error", "1.0538"],
]
t2 = Table(metrics_table, colWidths=[6*cm, 4*cm])
t2.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "DejaVuSans"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t2)
story.append(Spacer(1, 8))

story.append(Paragraph("MAE by moneyness bucket (moneyness = S0/K; puts are ITM when S0 &lt; K):", styles["BodyDV"]))
bucket_table = [
    ["Bucket", "MAE", "n (test)"],
    ["Deep ITM (S0/K < 0.85)", "0.0967", "419"],
    ["Near ATM (0.85-1.15)", "0.1140", "332"],
    ["Deep OTM (S0/K > 1.15)", "0.0614", "449"],
]
t3 = Table(bucket_table, colWidths=[6.5*cm, 2.5*cm, 3*cm])
t3.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "DejaVuSans"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t3)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "Near-ATM contracts have the highest MAE (0.114). This is expected: near-the-money "
    "American puts have the most curvature in price with respect to S0 (highest gamma), "
    "so a fixed-capacity MLP has the hardest time fitting that region tightly.",
    styles["BodyDV"]))

story.append(PageBreak())

story.append(Image(f"{PLOTS}/scatter_pred_vs_binomial.png", width=10.5*cm, height=10.5*cm))
story.append(Paragraph(
    "Figure 2. Predicted (NN) vs ground-truth (binomial) price on the test set. Points "
    "hug the y=x line closely across the full $0-$100 price range, with the visible "
    "scatter concentrated at low prices (near-zero OTM contracts) and mild widening at "
    "higher prices.", styles["CaptionDV"]))

story.append(Image(f"{PLOTS}/surface_comparison.png", width=17*cm, height=4.8*cm))
story.append(Paragraph(
    "Figure 3. Put price surface for fixed K=100, r=5%, sigma=25%, varying S0 and T. "
    "The binomial and NN surfaces (left, middle) are visually indistinguishable; the "
    "error surface (right) shows deviations concentrated at long maturities and low "
    "S0, on the order of a few cents to ~$0.30, well below the surface's own price scale "
    "(up to ~$50).", styles["CaptionDV"]))

story.append(Paragraph("Finance sanity checks", styles["H2DV"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Non-negativity:</b> 119/1,200 test predictions (&asymp;10%) and 40/3,600 "
        "surface-grid points were slightly negative (worst: -$0.33). All violations occur "
        "for deep-OTM, low-volatility, short-to-medium-maturity puts whose true price is "
        "essentially $0 &mdash; the unconstrained linear output layer has no floor at "
        "zero, so it can dip slightly below it. This is a real limitation, not a rounding "
        "artifact.", styles["BulletDV"])),
    ListItem(Paragraph(
        "<b>Monotonicity in S0:</b> checked the NN price along S0 = 50 to 150 "
        "(K=100, T=1, r=5%, sigma=25%, 199 steps). <b>0/199 steps increased</b> &mdash; "
        "the learned price is strictly non-increasing in spot, matching the no-arbitrage "
        "property of a put.", styles["BulletDV"])),
    ListItem(Paragraph(
        "<b>Intrinsic value floor:</b> 8/1,200 test predictions fell more than $0.25 below "
        "intrinsic value max(K-S0, 0) (worst breach $1.05, on a deep-ITM, low-vol, "
        "long-maturity contract). These are true violations of the American-put "
        "early-exercise floor and represent the sharpest failure mode found.",
        styles["BulletDV"])),
], bulletType="bullet", start="circle"))

story.append(Paragraph("Reflection: where does the model perform worst, and why?", styles["H2DV"]))
story.append(Paragraph(
    "Average error looks small (MAE $0.09 against prices up to ~$100), but averages hide "
    "two concrete weak regions:", styles["BodyDV"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Near-zero-price deep-OTM puts (low vol, short T):</b> the model occasionally "
        "predicts small negative prices here. The MSE loss barely penalizes a few-cent "
        "miss on a $0.00 label, so the network has little incentive to clamp at zero. "
        "This is the majority of the non-negativity violations and would be easy to fix "
        "with a ReLU/softplus output layer or a post-hoc clip.", styles["BulletDV"])),
    ListItem(Paragraph(
        "<b>High-price, high-curvature contracts (deep ITM or near ATM, longer T, "
        "moderate-to-high sigma):</b> this is where the single worst error ($1.05) and "
        "most of the intrinsic-floor breaches occur. These contracts have the steepest "
        "price surface and the largest absolute price scale, so a fixed absolute error "
        "budget from MSE training is spent disproportionately on the many cheap OTM "
        "contracts rather than the few expensive ITM ones.", styles["BulletDV"])),
], bulletType="bullet", start="circle"))
story.append(Paragraph(
    "Overall the model is <b>financially sensible but not exact</b>: it strictly respects "
    "monotonicity in S0, and its surface matches the binomial surface within a few cents "
    "almost everywhere, but it is not safe to deploy unclamped &mdash; a small fraction "
    "of quotes would violate the zero-price and intrinsic-value floors that any real put "
    "must satisfy. A production version should either clip outputs to "
    "max(intrinsic, 0) post-hoc, use a non-negative output activation, or add a "
    "no-arbitrage penalty term to the loss.", styles["BodyDV"]))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
story.append(Paragraph("Reproducibility", styles["H1DV"]))
story.append(Paragraph(
    "Code and repo link: <b>[insert your GitHub/Drive link here before submitting]</b>. "
    "Run order from a clean environment:", styles["BodyDV"]))
story.append(Paragraph(
    "pip install torch numpy pandas scikit-learn matplotlib reportlab<br/>"
    "python generate_dataset.py &nbsp;# Part A: builds data/american_put_dataset.csv<br/>"
    "python train_mlp.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Part B: trains MLP, saves best checkpoint<br/>"
    "python evaluate.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Part C: metrics, scatter, surface, sanity checks<br/>"
    "python plot_learning_curve.py",
    styles["MonoDV"]))
story.append(Paragraph(
    "All randomness (data sampling, train/val/test split, PyTorch weight init, "
    "DataLoader shuffling) is controlled by a single seed (42). Files produced: "
    "<b>data/american_put_dataset.csv</b>, <b>data/test_split.csv</b>, "
    "<b>checkpoints/best_mlp.pt</b>, <b>checkpoints/scaler.json</b>, "
    "<b>checkpoints/history.json</b>, <b>checkpoints/test_metrics.json</b>, "
    "<b>checkpoints/sanity_results.json</b>, and the four plots in <b>plots/</b>.",
    styles["BodyDV"]))

doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                         topMargin=1.6*cm, bottomMargin=1.6*cm,
                         leftMargin=1.8*cm, rightMargin=1.8*cm,
                         title="Week 6 - Neural Network Pricer for American Puts")
doc.build(story)
print(f"Saved report -> {OUT_PATH}")
