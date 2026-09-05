"""Build the technical note `mPL_to_Shapley.pdf` (repo root).

Body font: DejaVu Serif, registered straight out of matplotlib's font directory.
Equations: rendered to PNG with matplotlib mathtext (dejavuserif, 300 dpi) and
scaled into reportlab tables, so no TeX installation is needed.

Run:  python paper/paper_figs.py   (once, to build paper/assets/)
      python paper/build_pdf.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "assets")
EQDIR = os.path.join(ASSETS, "eq")
OUT_PDF = os.path.join(ROOT, "mPL_to_Shapley.pdf")

PAGE_W, PAGE_H = A4
MARGIN = 17.5 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
DATE = "September 5, 2026"

# ----------------------------------------------------------------- fonts ----

def register_fonts():
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    faces = {
        "DejaVuSerif": "DejaVuSerif.ttf",
        "DejaVuSerif-Bold": "DejaVuSerif-Bold.ttf",
        "DejaVuSerif-Italic": "DejaVuSerif-Italic.ttf",
        "DejaVuSerif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
    }
    for name, fn in faces.items():
        pdfmetrics.registerFont(TTFont(name, os.path.join(d, fn)))
    pdfmetrics.registerFontFamily(
        "DejaVuSerif", normal="DejaVuSerif", bold="DejaVuSerif-Bold",
        italic="DejaVuSerif-Italic", boldItalic="DejaVuSerif-BoldItalic",
    )


# ------------------------------------------------------------- equations ----

EQUATIONS = {
    "belief": r"$\mathrm{Pr}_{\theta}(x \mid I) \;=\; "
              r"\frac{\exp\left(s_{\theta}(x;I)/\tau\right)}"
              r"{\sum_{x' \in \mathcal{X}} \exp\left(s_{\theta}(x';I)/\tau\right)}$",
    "mpl_pair": r"$\mathrm{mPL}_{i,j}(S) \;=\; \frac{1000}{d_{\mathcal{X}}(x_i,x_j)}\,"
                r"\left|\, \log\frac{\mathrm{Pr}_{\theta}(x_i \mid I)}"
                r"{\mathrm{Pr}_{\theta}(x_j \mid I)} \;-\; "
                r"\log\frac{\mathrm{Pr}_{\theta}(x_i \mid I \ominus S)}"
                r"{\mathrm{Pr}_{\theta}(x_j \mid I \ominus S)} \right|$",
    "mpl_img": r"$\mathrm{mPL}(S) \;=\; \frac{1}{|\mathcal{P}|} "
               r"\sum_{(i,j) \in \mathcal{P}} \mathrm{mPL}_{i,j}(S)$",
    "tau": r"$\mathrm{mPL}^{(\tau)}(S) \;=\; \frac{1}{\tau}\, \mathrm{mPL}^{(1)}(S)$",
    "setfun": r"$v(S) \;=\; \mathrm{mPL}(S), \qquad v(\varnothing) = 0, "
              r"\qquad S \subseteq N$",
    "shapley": r"$\varphi_k \;=\; \sum_{S \subseteq N \setminus \{k\}} "
               r"\frac{|S|!\,(m - |S| - 1)!}{m!} \left[\, v(S \cup \{k\}) - v(S) \,\right]$",
    "efficiency": r"$\sum_{k=1}^{m} \varphi_k \;=\; v(N)$",
    "twocue": r"$\varphi_1 \;=\; \frac{1}{2} \left[\, v(\{1\}) \;+\; "
              r"\left( v(N) - v(\{2\}) \right) \right]$",
    "sii": r"$I^{\mathrm{SII}}(k,l) \;=\; \sum_{S \subseteq N \setminus \{k,l\}} "
           r"\frac{|S|!\,(m - |S| - 2)!}{(m-1)!} \left[\, v(S \cup \{k,l\}) "
           r"- v(S \cup \{k\}) - v(S \cup \{l\}) + v(S) \,\right]$",
    "dividends": r"$d_k \;=\; v(\{k\}), \qquad d_{kl} \;=\; v(\{k,l\}) - v(\{k\}) - v(\{l\})$",
    "order2": r"$\varphi^{(2)}_k \;=\; d_k + \frac{1}{2} \sum_{l \neq k} d_{kl}, "
              r"\qquad \varphi_k \;=\; \varphi^{(2)}_k + "
              r"\frac{v(N) - \sum_j \varphi^{(2)}_j}{m}$",
}


def render_equations(fontsize=13.0):
    """mathtext -> tight PNG at 300 dpi; returns {name: (path, w_pt, h_pt)}."""
    os.makedirs(EQDIR, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    out = {}
    for name, tex in EQUATIONS.items():
        path = os.path.join(EQDIR, f"{name}.png")
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=fontsize, color="black")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        plt.close(fig)
        w, h = PILImage.open(path).size
        out[name] = (path, w / 300.0 * 72.0, h / 300.0 * 72.0)
    return out


EQ = {}
EQ_NUM = {}


def eq_block(name, number, max_w=None):
    """Centered equation with a right-aligned number, as a 2-column table."""
    path, w, h = EQ[name]
    EQ_NUM[name] = number
    limit = max_w or (CONTENT_W - 40)
    if w > limit:
        h *= limit / w
        w = limit
    img = Image(path, width=w, height=h)
    t = Table([[img, Paragraph(f"({number})", STY["eqnum"])]],
              colWidths=[CONTENT_W - 34, 34])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------- styles ----

def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontName="DejaVuSerif-Bold", fontSize=17.5,
        leading=22.5, spaceAfter=8, alignment=TA_CENTER)
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=10.5,
        leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
    s["author"] = ParagraphStyle(
        "author", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=10.5,
        leading=14, alignment=TA_CENTER, spaceAfter=13)
    s["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=9.0,
        leading=12.4, alignment=TA_JUSTIFY, spaceAfter=4.4)
    s["abstract"] = ParagraphStyle(
        "abstract", parent=s["body"], fontSize=8.7, leading=12.0,
        leftIndent=15, rightIndent=15, spaceAfter=9)
    s["h1"] = ParagraphStyle(
        "h1", parent=ss["Normal"], fontName="DejaVuSerif-Bold", fontSize=12.0,
        leading=15.0, spaceBefore=7, spaceAfter=3.5)
    s["h2"] = ParagraphStyle(
        "h2", parent=ss["Normal"], fontName="DejaVuSerif-Bold", fontSize=10.3,
        leading=13.4, spaceBefore=7, spaceAfter=2.5)
    s["defbody"] = ParagraphStyle(
        "defbody", parent=s["body"], fontSize=8.9, leading=12.2, spaceAfter=0,
        alignment=TA_JUSTIFY)
    s["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=7.8,
        leading=10.2, alignment=TA_JUSTIFY, spaceBefore=3.0, spaceAfter=6,
        textColor=colors.HexColor("#2b2b2b"))
    s["eqnum"] = ParagraphStyle(
        "eqnum", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=9.6,
        leading=12, alignment=2)
    s["cell"] = ParagraphStyle(
        "cell", parent=ss["Normal"], fontName="DejaVuSerif", fontSize=8.0,
        leading=10.3)
    s["cellb"] = ParagraphStyle(
        "cellb", parent=s["cell"], fontName="DejaVuSerif-Bold")
    s["ref"] = ParagraphStyle(
        "ref", parent=s["body"], fontSize=8.4, leading=11.2, leftIndent=16,
        firstLineIndent=-16, spaceAfter=3, alignment=0)
    return s


STY = {}


# ----------------------------------------------------------- content bits ---

# DejaVu Serif has no U+1D4B3 / U+1D4AB, so the script metric-space and pair-set
# symbols are written as plain italic capitals, exactly as the previous note did.
SUBST = [
    ("&#119987;", "X"),     # script X  (metric space)
    ("&#119927;", "P"),     # script P  (admissible pair set)
    ("&#8201;%", "%"),      # no thin space before a percent sign
    ("&#8201;", "&nbsp;"),  # thin space -> non-breaking space (DejaVu drops U+2009)
]


def _fix(text):
    for a, b in SUBST:
        text = text.replace(a, b)
    return text


def P(text, style="body"):
    return Paragraph(_fix(text), STY[style])


def H1(text):
    return Paragraph(_fix(text), STY["h1"])


def H2(text):
    return Paragraph(_fix(text), STY["h2"])


def defbox(title, body):
    """The definition-box style: light panel with a rule, bold lead-in."""
    inner = Paragraph(_fix(f"<b>{title}</b>&nbsp; {body}"), STY["defbody"])
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f5f8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#8b98a8")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, colors.HexColor("#3b4a5a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def figure_parts(fname, caption, max_h=264, max_w=None):
    """[image, caption] — plain flowables, so they can be nested in a
    KeepTogether by the caller.  Never wrap a KeepTogether inside another one:
    reportlab's KeepTogether.wrap() reports height 0xffffff to force a split,
    which makes the outer one always overflow its page."""
    path = os.path.join(ASSETS, fname)
    pw, ph = PILImage.open(path).size
    w = max_w or CONTENT_W
    h = w * ph / pw
    if h > max_h:
        h = max_h
        w = h * pw / ph
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return [img, Paragraph(_fix(caption), STY["caption"])]


def figure(fname, caption, max_h=264, max_w=None):
    """A figure whose caption can never drift onto the next page without it."""
    return [Spacer(1, 3),
            KeepTogether(figure_parts(fname, caption, max_h, max_w))]


def table(rows, col_widths, caption, header=True, align_right=()):
    data = []
    for i, row in enumerate(rows):
        style = "cellb" if (header and i == 0) else "cell"
        data.append([Paragraph(_fix(str(c)), STY[style]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#3b4a5a")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#3b4a5a")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#3b4a5a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f7f9")]),
    ]
    for c in align_right:
        cmds.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    t.hAlign = "CENTER"
    return [Spacer(1, 3), t, Paragraph(_fix(caption), STY["caption"])]


# ------------------------------------------------------------------ story ---

def build_story():
    S = []

    S.append(P("From mPL to Shapley Attribution:<br/>"
               "Quantifying Per-Cue Location-Privacy Leakage in Images", "title"))
    S.append(P("Technical note &mdash; GeoBayes project", "subtitle"))
    S.append(P(f"Xinpeng Xie &middot; {DATE}", "author"))

    S.append(P(
        "<b>Abstract.</b> A photograph leaks its capture location through many visual cues. "
        "We quantify how much each cue leaks with respect to a concrete adversary. Leakage of "
        "a cue subset is measured by mPL, a distance-normalized posterior-leakage metric. "
        "Because masking effects are non-additive (cues overlap or back each other up), "
        "single-cue mPL is a biased per-cue estimate; we treat leakage as a set function and "
        "attribute it with the Shapley value, whose efficiency axiom makes per-cue shares sum "
        "exactly to the total removable leakage of the identified cue set. The pipeline "
        "(GPT-4o cue naming, SAM 3 masks, GeoRanker adversary) is evaluated on 100 im2gps3k "
        "images. This revision adds three components. The cue inventory is de-duplicated "
        "geometrically before attribution, collapsing 244 named cues into 223 players and "
        "raising the median attributed leakage of text/signage from 0.080 to 0.117 without "
        "changing the category ranking. The removal operator is upgraded from gray fill to "
        "LaMa inpainting, which halves the masking-artifact floor (median control mPL 0.087 "
        "&rarr; 0.045) and raises the fraction of cues that resolve above their own artifact "
        "null from 48.8&#8201;% to 76.6&#8201;%, while leaving the ranking and the "
        "non-additivity intact. And a second-order anchored Shapley estimator makes the "
        "attribution computable from <i>m</i>&#8201;+&#8201;C(<i>m</i>,2)&#8201;+&#8201;1 "
        "scored subsets instead of 2<super>m</super>, exactly for <i>m</i>&#8201;&#8804;&#8201;3 "
        "and with rank correlation 1.000 beyond it.", "abstract"))

    # ---------------------------------------------------------------- §1 ----
    S.append(KeepTogether([
        H1("1&nbsp;&nbsp;Problem Statement"),
        P("Let <i>I</i> be an image with <i>m</i> identified cues "
          "<i>N</i> = {1, &hellip;, <i>m</i>}, each with a pixel mask, and an adversary who "
          "forms a posterior over a fixed candidate set. Two questions: how much belief "
          "movement is caused by masking any cue subset (measurement, &sect;2), and how to "
          "divide the total leakage among individual cues when effects do not add up "
          "(attribution, &sect;4). Two further questions decide whether the answer is an "
          "artifact of the measurement rather than a property of the image: who is entitled to "
          "name a cue (&sect;6), and what &ldquo;masking&rdquo; should physically mean "
          "(&sect;7)."),
    ]))

    # ---------------------------------------------------------------- §2 ----
    S.append(KeepTogether([
        H1("2&nbsp;&nbsp;Privacy Criterion and Adversary Model"),
        H2("2.1&nbsp;&nbsp;Setup"),
        P("Let <i>X</i> &isin; <i>&#119987;</i> denote the capture location of image <i>I</i>, "
          "where (<i>&#119987;</i>, <i>d</i><sub><i>&#119987;</i></sub>) is a finite metric "
          "space of candidate locations and <i>d</i><sub><i>&#119987;</i></sub> is the "
          "haversine distance. We instantiate <i>&#119987;</i> with the <i>n</i> = 138 "
          "city-level ground-truth labels of our im2gps3k subset (administrative aliases "
          "within 2&#8201;km merged &mdash; only City of Westminster / Greater London, "
          "1.3&#8201;km &mdash; leaving 137 points; genuinely distinct nearby places are kept "
          "separate). For a cue subset <i>S</i> &sube; <i>N</i>, the masked image "
          "<i>I</i>&#8201;&#8854;&#8201;<i>S</i> &mdash; removing the union of the SAM 3 masks "
          "of the cues in <i>S</i> &mdash; plays the role of the sanitized release. Which "
          "removal operator implements &#8854; is a modelling choice and not a detail: we take "
          "LaMa inpainting as the primary operator and gray fill as a robustness variant, and "
          "&sect;7 reports what changes between them."),
    ]))
    S.append(P(
        "<b>Adversary model.</b> The criterion below is adversary-agnostic; for empirical "
        "auditing we instantiate the adversary as a neural posterior estimator with scoring "
        "function <i>s</i><sub>&theta;</sub>(<i>x</i>;&#8201;<i>I</i>) &mdash; the GeoRanker "
        "reward model (Qwen2-VL-7B + LoRA + value head), whose score increases with geographic "
        "closeness between the depicted scene and candidate <i>x</i>. Combining the scores by a "
        "softmax yields a normalized adversarial belief (we avoid calling it a calibrated "
        "Bayesian posterior; see &sect;2.3), with &tau; = 1 and deterministic decoding at fixed "
        "input:"))
    S.append(eq_block("belief", 1))

    S.append(H2("2.2&nbsp;&nbsp;Metric-normalized posterior leakage"))
    S.append(defbox(
        "Definition 2.1 (mPL, following [4]).",
        "For <i>x<sub>i</sub></i>, <i>x<sub>j</sub></i> &isin; <i>&#119987;</i> with "
        "<i>x<sub>i</sub></i> &ne; <i>x<sub>j</sub></i> and "
        "<i>d</i><sub><i>&#119987;</i></sub>(<i>x<sub>i</sub></i>, <i>x<sub>j</sub></i>) &gt; 0, "
        "the pairwise leakage of cue subset <i>S</i> is given by equation (2), and the "
        "image-level leakage aggregates it over the admissible pair set <i>&#119927;</i> "
        "(pairs with positive posterior mass under both releases) as in equation (3)."))
    S.append(eq_block("mpl_pair", 2))
    S.append(eq_block("mpl_img", 3))
    S.append(P(
        "Here Pr<sub>&theta;</sub>(&middot;&#8201;|&#8201;<i>I</i>) is the full-view belief and "
        "Pr<sub>&theta;</sub>(&middot;&#8201;|&#8201;<i>I</i>&#8201;&#8854;&#8201;<i>S</i>) the "
        "counterfactual baseline belief (masked view): mPL measures the distance-normalized "
        "odds shift between them; the factor 1000 sets the units to nats per 1000&#8201;km. The "
        "absolute value makes mPL direction-agnostic. Note <i>v</i>(<i>N</i>), masking all "
        "identified cues, is not the total leakage of the image &mdash; residual background "
        "(climate, terrain, road layout) can still localize; <i>v</i>(<i>N</i>) is the total "
        "<i>removable</i> leakage of the identified cue set, and &sect;6 shows how much that "
        "set depends on who names the cues."))

    S.append(H2("2.3&nbsp;&nbsp;Calibration and temperature invariance"))
    S.append(P(
        "The GeoRanker score is a ranking/distance reward, not a probability-trained "
        "log-likelihood, so the softmax is a belief, not a calibrated posterior. This matters "
        "because mPL scales inversely with &tau;: since <i>s</i> cancels additively in the "
        "log-odds,"))
    S.append(eq_block("tau", 4))
    S.append(P(
        "verified exactly on our data (e.g. mPL &times;2.00 / &times;0.50 / &times;0.20 at "
        "&tau; = 0.5 / 2 / 5). Two consequences. (i) &tau; is a pure global scale, so "
        "within-image subset rankings and interaction signs &mdash; on which all our "
        "conclusions rest &mdash; are exactly &tau;-invariant; absolute magnitudes are reported "
        "in &tau; = 1 units. (ii) For the absolute scale, temperature scaling can be fit on "
        "held-out true-label NLL: on a 47/48 split the optimum is &tau;* = 0.95 (near 1), with "
        "test NLL 2.21&#8201;&rarr;&#8201;2.19 and Brier 0.643&#8201;&rarr;&#8201;0.639, "
        "confirming &tau; = 1 is already near-optimal. We therefore report a score-induced "
        "belief-shift attribution rather than a strictly Bayesian posterior-leakage guarantee."))

    # ---------------------------------------------------------------- §3 ----
    S.append(KeepTogether([
        H1("3&nbsp;&nbsp;Non-Additivity: Leakage as a Set Function"),
        P("Define the leakage set function on cue subsets:"),
    ]))
    S.append(eq_block("setfun", 5))
    S.append(P(
        "Empirically <i>v</i>(<i>S</i>&#8201;&cup;&#8201;<i>T</i>) &ne; <i>v</i>(<i>S</i>) + "
        "<i>v</i>(<i>T</i>), in both directions. <b>Backup</b> (super-additive): redundant cues "
        "compensate for each other, so <i>v</i>({<i>k</i>}) &asymp; 0 for every <i>k</i> while "
        "<i>v</i>(<i>N</i>) is large (an Okazaki festival scene shows exactly this). "
        "<b>Overlap</b> (sub-additive): cues carrying overlapping evidence each cause most of "
        "the movement alone, so <i>v</i>({1,&#8201;2}) &lt; <i>v</i>({1}) + <i>v</i>({2}) "
        "(limestone cliffs vs. rocky shoreline in a Capri photo). Consequently no single "
        "masking condition yields a fair per-cue number: the leave-one-out marginal "
        "<i>v</i>(<i>N</i>) &minus; <i>v</i>(<i>N</i>&#8201;\\&#8201;{<i>k</i>}) under-credits "
        "backup cues, while the leave-one-in value <i>v</i>({<i>k</i>}) over-credits "
        "overlapping ones; in our experiments the two disagreed by up to an order of magnitude. "
        "Sub-additivity dominates and is not an artifact of the removal operator: under LaMa "
        "inpainting 83.8&#8201;% of the 80 multi-cue images are sub-additive, with median "
        "<i>v</i>(<i>N</i>)&#8201;/&#8201;&Sigma;<sub><i>k</i></sub> <i>v</i>({<i>k</i>}) = "
        "0.621 (&sect;7)."))

    # ---------------------------------------------------------------- §4 ----
    S.append(KeepTogether([
        H1("4&nbsp;&nbsp;Shapley Attribution"),
        defbox("Definition 4.1 (Shapley-mPL).",
               "The attributed leakage of cue <i>k</i> is its Shapley value under the set "
               "function <i>v</i>, equation (6)."),
    ]))
    S.append(eq_block("shapley", 6))
    S.append(P(
        "&phi;<sub><i>k</i></sub> is the average marginal damage of removing cue <i>k</i> over "
        "all <i>m</i>! removal orders, and the unique attribution satisfying efficiency, "
        "symmetry, null player, and linearity. Efficiency is the property we rely on:"))
    S.append(eq_block("efficiency", 7))
    S.append(P(
        "so &ldquo;cue <i>k</i> contributes &phi;<sub><i>k</i></sub>&#8201;/&#8201;"
        "<i>v</i>(<i>N</i>) of the leakage&rdquo; is well defined. For two-cue images (the "
        "median case) the formula collapses to the average of the leave-one-in and "
        "leave-one-out readings:"))
    S.append(eq_block("twocue", 8))
    S.append(P(
        "Pairwise structure is captured by the Shapley interaction index (SII), which averages "
        "the second-order difference over all coalition contexts:"))
    S.append(eq_block("sii", 9))
    S.append(P(
        "with <i>I</i><super>SII</super> &lt; 0 = overlap and &gt; 0 = backup. <b>Worked "
        "example (Capri).</b> <i>v</i>({cliffs}) = 0.088, <i>v</i>({shoreline}) = 0.104, "
        "<i>v</i>(<i>N</i>) = 0.113 give &phi;<sub>cliffs</sub> = 0.049, "
        "&phi;<sub>shoreline</sub> = 0.065 (sum = <i>v</i>(<i>N</i>)), and "
        "<i>I</i><super>SII</super> = &minus;0.079 flags the two coastal cues as overlapping. "
        "&phi;<sub><i>k</i></sub> can legitimately be negative (masking <i>k</i> pushes belief "
        "back); such values are reported as measured."))

    S.append(KeepTogether([
        H2("4.1&nbsp;&nbsp;Second-order anchored Shapley: the computational shortcut"),
        P("The exact value in (6) needs all 2<super><i>m</i></super> subsets. Where the lattice "
          "is not fully scored &mdash; the inpainting arm of &sect;7 and the vocabulary "
          "ablation of &sect;6 evaluate only singletons, pairs and the grand coalition &mdash; "
          "we use the second-order truncation of the Harsanyi decomposition. Writing the "
          "first- and second-order dividends"),
    ]))
    S.append(eq_block("dividends", 10))
    S.append(P("the truncated value and its anchored correction are"))
    S.append(eq_block("order2", 11))
    S.append(P(
        "The anchoring term restores efficiency by construction, so "
        "&Sigma;<sub><i>k</i></sub> &phi;<sub><i>k</i></sub> = <i>v</i>(<i>N</i>) still holds "
        "exactly. The cost is <i>m</i> + C(<i>m</i>,&#8201;2) + 1 scored subsets instead of "
        "2<super><i>m</i></super> &minus; 1 &mdash; 16 instead of 31 at <i>m</i> = 5, 37 "
        "instead of 255 at <i>m</i> = 8."))
    S.append(P(
        "<b>Validation.</b> Against the exact gray lattice on all 80 multi-cue images, the "
        "estimator is exact for <i>m</i> &le; 3 (31 images with <i>m</i> = 2, 30 with "
        "<i>m</i> = 3; median relative error at machine precision, &lt; "
        "2&#8201;&times;&#8201;10<super>&minus;16</super>). At <i>m</i> = 4 (18 images) the "
        "median magnitude error is 7.7&#8201;% and the maximum 10.1&#8201;%; the single "
        "<i>m</i> = 5 image is off by 54&#8201;%. Ranking is unaffected throughout: the "
        "within-image Spearman correlation between &phi;<super>(2)</super> and the exact "
        "&phi; has median 1.000 and the top-1 cue agrees on 100&#8201;% of the 80 images. The "
        "median truncation residual "
        "|<i>v</i>(<i>N</i>)&#8201;&minus;&#8201;&Sigma;&phi;<super>(2)</super>|&#8201;/&#8201;"
        "|<i>v</i>(<i>N</i>)| is 0.031. We therefore read order-2 magnitudes as approximate for "
        "<i>m</i> &ge; 4 and as a ranking only for <i>m</i> &ge; 5."))

    # ---------------------------------------------------------------- §5 ----
    S.append(KeepTogether(
        [H1("5&nbsp;&nbsp;Experimental Pipeline")]
        + figure_parts("fig1_pipeline.png",
               "<b>Figure 1: Pipeline.</b> Cues are named by GPT-4o, localized by SAM 3 and "
               "merged when their masks coincide; every subset is removed by LaMa inpainting "
               "(gray fill as a robustness variant) and scored by the GeoRanker belief over the "
               "138-label metric space; mPL over the subset lattice defines <i>v</i>, which the "
               "Shapley value attributes to individual players, and equal-area control "
               "placements supply the artifact null a cue must clear to count as sensitive.",
               max_h=158)))
    S.extend(table(
        [["Component", "Choice"],
         ["Dataset", "im2gps3k subset, 100 hi-res images (95 with maskable cues)"],
         ["Cue naming", "GPT-4o grounded reasoning (name, category, segment query)"],
         ["Segmentation", "SAM 3 text-to-mask; degenerate masks (&gt;40&#8201;% of frame) excluded"],
         ["De-duplication", "union-find over mask IoU &ge; 0.9; 244 cues &rarr; 223 players "
                            "(14 groups, 14 images)"],
         ["Removal operator", "LaMa inpainting (primary); gray fill (robustness variant)"],
         ["Metric space", "138 GT labels + GPS, haversine; 2&#8201;km alias dedup (137 points)"],
         ["Adversary", "GeoRanker: Qwen2-VL-7B + LoRA + value head, nf4, deterministic"],
         ["Elicitation", "prompt = image + candidate GPS + label text; softmax &tau; = 1"],
         ["Metric", "mPL (nats/1000&#8201;km), within-image; full posteriors persisted"],
         ["Attribution", "Shapley &phi; + interaction <i>I</i><super>SII</super>; exact "
                         "2<super><i>m</i></super> lattice for <i>m</i> &le; 5, order-2 "
                         "anchored otherwise"],
         ["Artifact null", "equal-area control placements: 488 inpainted, 516 gray"]],
        [95, CONTENT_W - 95],
        "<b>Table 1: Implementation summary.</b>"))

    # ---------------------------------------------------------------- §6 ----
    S.append(KeepTogether([
        H1("6&nbsp;&nbsp;Cue Inventory: De-duplication and a Fixed-Vocabulary Ablation"),
        H2("6.1&nbsp;&nbsp;Geometric de-duplication"),
        P("GPT-4o names cues in language; SAM 3 grounds them in pixels. Nothing forces two "
          "different names to land on different pixels, and frequently they do not: "
          "&ldquo;Thai text on storefront signs&rdquo; and &ldquo;Fujifilm signage&rdquo; are "
          "one and the same region of one Bangkok photo. Two players whose masks coincide split "
          "a single contribution, and the Shapley value &mdash; correctly, given the inputs "
          "&mdash; pays each of them half. The inventory, not the attribution, is at fault."),
    ]))
    S.append(P(
        "We merge cues by union-find over pairwise mask IoU &ge; 0.9, computed on exactly the "
        "per-cue union masks the masking code uses. Containment (recall &ge; 0.95 with IoU below "
        "the merge threshold) is reported but never merged &mdash; 4 such pairs survive, "
        "including the Capri limestone cliffs sitting inside the rocky shoreline. Across the 95 "
        "maskable images this collapses <b>244 named cues into 223 players</b>, affecting 14 "
        "images through 14 merged groups (21 duplicates removed). No re-scoring is required: "
        "the members of a merged group cover near-identical pixels, so the masked image of a "
        "merged subset equals that of the union of its members&rsquo; originals and "
        "<i>v</i>&prime; is read straight off the existing complete gray lattice. 12 of the 14 "
        "groups are pixel-exact; the remaining two differ by 9.35&#8201;% and 0.23&#8201;% of "
        "their union area. That is the single approximation introduced by de-duplication."))
    S.append(P(
        "Credit consolidates as it should: &phi;(merged) &asymp; &Sigma;&phi;(members, before), "
        "with median |&Delta;| = 0.0063 and worst case 0.0357 (25.1&#8201;% of that image&rsquo;s "
        "<i>v</i>(<i>N</i>)); the residual is negative in 9 of the 10 partial groups, which is "
        "the standard merging effect &mdash; a contribution either duplicate could have supplied "
        "is now paid once. Efficiency stays exact, max "
        "|&Sigma;&phi;<sub><i>k</i></sub> &minus; <i>v</i>(<i>N</i>)| = "
        "1.11&#8201;&times;&#8201;10<super>&minus;16</super> before and after. The effect on the "
        "headline is concentrated in one category: the median &phi; of text/signage rises from "
        "0.080 (<i>n</i> = 29) to <b>0.117</b> (<i>n</i> = 24), +45&#8201;%, because several "
        "images (Cambridge, Spa, Bardo, Bangkok, Paris) had two to four separately named signs "
        "segmented onto the same pixels. Every other category moves by at most 0.002 and the "
        "<b>category ranking is unchanged</b>. Pairwise interactions fall from 239 to 192 pairs: "
        "30 before-pairs were pure geometric duplicates, 29 of which had been counted as "
        "&ldquo;overlap&rdquo; (median <i>I</i><super>SII</super> = &minus;0.058), so the "
        "duplicate-driven overlap disappears with them. The Harsanyi mass moves toward first "
        "order accordingly."))
    S.extend(figure(
        "fig4_dedup.png",
        "<b>Figure 2: Geometric de-duplication.</b> (a) per-category median &phi; before vs "
        "after &mdash; only text/signage moves materially; (b) the SII distribution loses its "
        "duplicate-driven overlap mass; (c) credit consolidation for the 14 merged groups "
        "against <i>y</i> = <i>x</i>; (d) the merged groups themselves.",
        max_h=257))
    S.extend(table(
        [["", "cue pairs", "overlap (<i>I</i><super>SII</super> &lt; &minus;0.01)",
          "&asymp;&nbsp;add.", "backup (&gt; 0.01)", "median <i>I</i><super>SII</super>",
          "Harsanyi order 1/2/3/4 (%)"],
         ["before de-dup (244 cues)", "239", "182", "34", "23", "&minus;0.028",
          "60.5 / 32.1 / 13.2 / 3.8"],
         ["after de-dup (223 players)", "192", "138", "35", "19", "&minus;0.024",
          "65.5 / 29.7 / 11.6 / 3.2"]],
        [104, 36, 86, 44, 50, 50, CONTENT_W - 370],
        "<b>Table 2: What de-duplication removes.</b> 44 of the 182 overlap pairs vanish, 29 of "
        "them pure geometric duplicates. Harsanyi shares are medians of the |dividend| mass per "
        "order over multi-cue images (80 before, 76 after; the single <i>m</i> = 5 image becomes "
        "<i>m</i> = 3, so order 5 disappears)."))

    S.append(KeepTogether([
        H2("6.2&nbsp;&nbsp;Who is entitled to name a cue? A fixed-vocabulary ablation"),
        P("A top-down inventory inherits whatever a language model considers worth mentioning. "
          "To bound that dependence we re-ran 10 images with a bottom-up inventory: 12 generic "
          "SAM 3 concepts (text or signage, building facade, vehicle, person&rsquo;s clothing, "
          "tree or vegetation, mountain, road surface, street furniture, statue or monument, "
          "flag, &hellip;) applied uniformly, with the same removal operator (LaMa inpainting), "
          "the same adversary, and the order-2 anchored &phi; of &sect;4.1. Only the definition "
          "of &ldquo;a cue&rdquo; changes."),
    ]))
    S.append(P(
        "<b>Total removable leakage favours the top-down inventory.</b> Summed over the 10 "
        "images, <i>v</i>(<i>N</i>) is <b>4.93</b> for the 28 GPT-4o cues against <b>3.50</b> "
        "for the 33 vocabulary cues (0.71&times;); GPT-4o wins 8 images, the vocabulary 1 "
        "(Bangkok, 1.73&times;, where seven generic regions beat three named ones), with 1 exact "
        "tie (Seville, where both reduce to the same statue). The median per-image ratio is "
        "0.83&times;."))
    S.extend(figure(
        "fig5_vocab.png",
        "<b>Figure 3: Top-down vs bottom-up cue inventories</b> (10 images, LaMa inpainting, "
        "GeoRanker). (a) total removable leakage per image; (b) per-cue &phi; with the "
        "per-image artifact null, matched regions joined by a line; (c) the two unmatched "
        "populations &mdash; the vocabulary&rsquo;s exclusive finds are numerous but weak, "
        "GPT-4o&rsquo;s are few but strong.",
        max_h=232))
    S.append(P(
        "<b>What a fixed vocabulary systematically misses.</b> 7 of the 28 GPT-4o cues have no "
        "vocabulary counterpart at IoU &ge; 0.1, and 5 of those clear the artifact null: 4 "
        "landmarks/buildings, 2 architecture, 1 text/signage. Three gaps are structural rather "
        "than accidental. <i>Masonry and ruins</i>: at Tinum the vocabulary finds only a "
        "1.5&#8201;%-area &ldquo;tree or vegetation&rdquo; and reaches <i>v</i>(<i>N</i>) = "
        "0.029, against 0.689 for &ldquo;Mayan architectural style&rdquo; (&phi; = 0.328) and "
        "&ldquo;Ruined stone structures&rdquo; (&phi; = 0.366). <i>Ornamental architectural "
        "detail</i>: the Paris &ldquo;Golden clock with ornate design&rdquo; (&phi; = 0.027) is "
        "not any of the 12 concepts. <i>Text on portable objects</i>: the Chicago "
        "&ldquo;Wrigley Field text on cup&rdquo; (&phi; = 0.472) carries that image&rsquo;s "
        "entire leakage and no generic concept covers a cup."))
    S.append(P(
        "<b>The converse gap is real but weak.</b> 11 of the 33 vocabulary cues have no GPT-4o "
        "counterpart and 8 of them are sensitive (73&#8201;%), but their &phi; median is 0.013 "
        "against 0.049 for matched cues &mdash; mostly generic clothing, road surface and street "
        "furniture. Two are not negligible: in Chicago, where GPT-4o named only the cup, "
        "&ldquo;person&rsquo;s clothing&rdquo; (&phi; = 0.098) and &ldquo;street furniture&rdquo; "
        "(&phi; = 0.077) are leaky regions the top-down inventory never proposed. Where both "
        "inventories do find the same region &mdash; 13 mutual-best pairs at IoU &ge; 0.5 "
        "&mdash; the two attributions agree: Spearman &rho; = 0.80, sensitive-flag agreement "
        "12/13."))
    S.append(P(
        "<b>Conclusion: neither inventory is a superset.</b> Merging regions across the two "
        "sides at IoU &ge; 0.5, the union of sensitive regions is <b>38</b> = 11 found by both "
        "+ 11 only by GPT-4o + 16 only by the fixed vocabulary. The right default is therefore "
        "a hybrid inventory &mdash; run both and merge geometrically, exactly as &sect;6.1 "
        "already does within one inventory. The 95-image results of &sect;8 use the GPT-4o side "
        "only and consequently under-count the generic-background leakage the vocabulary "
        "exposes; the ordering of the named cues is not affected."))

    # ---------------------------------------------------------------- §7 ----
    S.append(KeepTogether([
        H1("7&nbsp;&nbsp;Removal Operator and the Masking-Artifact Floor"),
        P("Gray-filling a mask does not only delete evidence, it also writes a flat gray blob "
          "into the image, and a vision model can react to the blob itself. Any belief movement "
          "produced that way is a masking artifact, not leakage. We therefore re-ran the entire "
          "measurement with LaMa inpainting, which fills the mask with plausible surrounding "
          "texture, and read the difference between the two arms as a test of operator "
          "dependence: 95 images, 578 scored variants (all singletons, all pairs, the grand "
          "coalition) plus 488 equal-area control placements."),
    ]))
    S.append(P(
        "<b>(a) The cue ordering survives.</b> Over the 244 paired cues, Spearman "
        "&rho;(gray, inpaint) = <b>0.856</b> (Pearson 0.879). Inpainting yields the smaller "
        "value for 57.4&#8201;% of cues and the median single-cue mPL falls from <b>0.098 to "
        "0.092</b> (median ratio 0.939&times;) &mdash; a level shift of about 6&#8201;%, not a "
        "reordering."))
    S.append(P(
        "<b>(b) Non-additivity is operator-independent.</b> Under inpainting 83.8&#8201;% of the "
        "80 multi-cue images are sub-additive with median "
        "<i>v</i>(<i>N</i>)&#8201;/&#8201;&Sigma;<sub><i>k</i></sub><i>v</i>({<i>k</i>}) = "
        "<b>0.621</b>, the same picture &sect;3 draws under gray fill. The empty-context second "
        "difference is more negative under inpainting (204 redundant / 18 &asymp;0 / 17 "
        "synergistic, against the gray SII&rsquo;s 182 / 34 / 23), but that gap is the known "
        "bias of the empty-context index &mdash; the gray empty-context index gives 215 / 12 / "
        "12 on the same images &mdash; not an operator effect."))
    S.append(P(
        "<b>(c) The artifact floor halves.</b> Each cue&rsquo;s equal-area control masks a "
        "region of the same area at a random non-overlapping position; its mPL is the movement "
        "the operator produces by itself. Inpainted controls move the belief far less than gray "
        "ones: median <b>0.045 vs 0.087</b>, P90 <b>0.131 vs 0.189</b>. This comparison is "
        "<i>unpaired</i> &mdash; the inpaint control run contains no same-placement gray twins, "
        "so the gray side comes from the earlier, independent run at different random positions "
        "(516 gray placements against 488 inpainted) and the statement is distribution-level, "
        "not per-placement. The consequence is nevertheless large: the fraction of cues whose "
        "real removal exceeds the strongest of its own controls &mdash; the resolvable fraction "
        "&mdash; rises from <b>48.8&#8201;% to 76.6&#8201;%</b> (<i>n</i> = 244 each), and the "
        "gain is biggest exactly where gray fill was least trustworthy (Table 3). After "
        "de-duplication both arms are essentially unchanged: gray 47.1&#8201;% (<i>n</i> = 223), "
        "inpaint 76.0&#8201;% (<i>n</i> = 221; two merged players group three originals whose "
        "union was never scored in the inpaint arm)."))
    S.extend(table(
        [["Category", "<i>n</i>", "resolvable, gray", "resolvable, inpaint"],
         ["road/infrastructure", "10", "40.0&#8201;%", "100.0&#8201;%"],
         ["text/signage", "29", "65.5&#8201;%", "89.7&#8201;%"],
         ["landmarks/buildings", "31", "51.6&#8201;%", "83.9&#8201;%"],
         ["commercial/cultural", "42", "54.8&#8201;%", "78.6&#8201;%"],
         ["vehicles/license plates", "12", "66.7&#8201;%", "75.0&#8201;%"],
         ["environment", "78", "39.7&#8201;%", "69.2&#8201;%"],
         ["architecture", "41", "43.9&#8201;%", "68.3&#8201;%"],
         ["other", "1", "0.0&#8201;%", "100.0&#8201;%"],
         ["<b>all cues</b>", "<b>244</b>", "<b>48.8&#8201;%</b>", "<b>76.6&#8201;%</b>"]],
        [160, 40, 120, 120],
        "<b>Table 3: Resolvability above the artifact floor</b> (real removal &gt; max of that "
        "cue&rsquo;s own equal-area controls). Unpaired: the gray column is the earlier control "
        "run at different random placements."))
    S.append(P(
        "<b>(d) What changes in the attribution.</b> The order-2 anchored &phi; under "
        "inpainting correlates with the exact gray &phi; at &rho; = 0.750 globally, with "
        "within-image median 1.000 and 72.5&#8201;% top-1 agreement; the median &phi; barely "
        "moves (0.055&#8201;&rarr;&#8201;0.056). One category moves materially: vehicles/license "
        "plates, from 0.045 to 0.075. A plausible reading &mdash; which we state as a "
        "<i>hypothesis</i>, not a result &mdash; is that gray fill leaves the vehicle&rsquo;s "
        "silhouette and its placement in the scene legible, so part of the cue survives the "
        "removal it was supposed to undergo, whereas inpainting removes the silhouette too and "
        "the measured contribution rises accordingly. Testing it would require a "
        "silhouette-preserving inpainting control, which we have not run."))
    S.extend(figure(
        "fig6_inpaint.png",
        "<b>Figure 4: Inpainting vs gray-fill masking.</b> (a) single-cue agreement, "
        "&rho; = 0.856; (b) per-category attribution, gray exact &phi; vs inpaint order-2 &phi;; "
        "(c) the interaction distribution under both operators; (d) the artifact floor and the "
        "resolvable fraction &mdash; note that the two control populations are unpaired.",
        max_h=245))

    # ---------------------------------------------------------------- §8 ----
    S.append(KeepTogether([
        H1("8&nbsp;&nbsp;Results on 100 Images"),
        P("Across the 95 images with maskable cues the adversary attains a country-scale "
          "accuracy (error &lt; 750&#8201;km) of 77.9&#8201;%, 45.3&#8201;% within 25&#8201;km, "
          "median error 59&#8201;km (median <i>p</i><sub>true</sub> = 0.137). Attribution is "
          "computed on all 95 images (the primary result) over the de-duplicated inventory of "
          "223 players; we additionally stratify into 71 country-correct and 24 "
          "country-incorrect images, since a wrong top-1 does not imply zero belief movement "
          "(mPL measures odds shift, not top-1 correctness). The numbers below are the exact "
          "gray-lattice values &mdash; the only fully evaluated 2<super><i>m</i></super> arm "
          "&mdash; retained as the operator-robustness variant of the inpainting pipeline of "
          "&sect;7, whose within-image ranking they reproduce (median &rho; = 1.000)."),
    ]))
    S.extend(figure(
        "fig2_sweep.png",
        "<b>Figure 5:</b> (a) adversary accuracy at increasing error thresholds; (b) confidence "
        "on the true label; (c) per-category single-cue mPL (median) on the original 244-cue "
        "inventory &mdash; the traditional ablation baseline; see Figure 6 for the "
        "de-duplicated, Shapley-corrected attribution.",
        max_h=157))
    S.append(P(
        "<b>Per-category leakage.</b> As the traditional ablation baseline, the raw single-cue "
        "masking effect on the de-duplicated inventory (median mPL, all 95 images) ranks "
        "text/signage first (0.133, <i>n</i> = 24), then landmarks/buildings 0.120 (28), "
        "vehicles/plates 0.111 (11), environment 0.096 (76), architecture 0.086 (40), "
        "road/infrastructure 0.080 (10), commercial/cultural 0.074 (33). By &sect;3 this "
        "over-states per-cue attribution by double-counting shared leakage. Our primary result "
        "is the Shapley-corrected ranking (Figure 6b): <b>text/signage 0.117</b> &gt; "
        "landmarks/buildings 0.072 &gt; environment 0.061 &gt; vehicles/plates 0.046 &gt; "
        "architecture 0.043 &gt; commercial/cultural 0.037 &gt; road/infrastructure 0.034. The "
        "correction removes 40&ndash;60&#8201;% of every category&rsquo;s median but only "
        "12&#8201;% of text/signage&rsquo;s, and reorders the middle: vehicles/plates falls from "
        "3rd (raw) to 4th &mdash; its apparent standalone importance is largely shared with "
        "co-occurring cues in the same scene &mdash; whereas text/signage remains 1st by a "
        "widening margin, i.e. genuinely the least-shared leakage source. (Restricting to the 71 "
        "country-correct images leaves the top three intact but pushes vehicles/plates from 4th "
        "to 6th on <i>n</i> = 6, confirming that country-conditioning is itself a selection "
        "effect; we therefore keep all 95 as the primary set.)"))
    S.append(P(
        "<b>Shapley attribution.</b> The full 2<super><i>m</i></super> lattice is evaluated for "
        "every multi-cue image (<i>m</i> = 2&ndash;5 before de-duplication, 2&ndash;4 after) and "
        "augmented with the single-player images (&phi; = <i>v</i>({1}), 19 of 95), so all 95 "
        "images receive an attribution; the efficiency identity "
        "&Sigma;&phi;<sub><i>k</i></sub> = <i>v</i>(<i>N</i>) holds exactly on the 76 images "
        "with <i>M</i> &ge; 2, to 1.11&#8201;&times;&#8201;10<super>&minus;16</super>. "
        "Interactions use the Shapley interaction index of &sect;4 (averaged over coalition "
        "contexts), not the empty-context second difference. Across the 192 de-duplicated cue "
        "pairs (Table 4, Figure 2b):"))
    S.extend(table(
        [["Interaction type", "Condition", "Pairs", "Meaning"],
         ["Overlap (sub-additive)", "<i>I</i><super>SII</super> &lt; &minus;0.01", "138",
          "either cue alone does most of the movement"],
         ["&asymp; Additive", "|<i>I</i><super>SII</super>| &le; 0.01", "35",
          "independent contributions"],
         ["Backup (super-additive)", "<i>I</i><super>SII</super> &gt; 0.01", "19",
          "cues must be masked jointly"]],
        [118, 96, 38, CONTENT_W - 252],
        "<b>Table 4: Shapley-interaction structure across the 192 pairs</b> of the "
        "de-duplicated inventory (all maskable images; median <i>I</i><super>SII</super> = "
        "&minus;0.024). The empty-context second difference over-states overlap on the same "
        "pairs (171 / 12 / 9)."))
    S.append(P(
        "Overlap dominance (138/192, 72&#8201;% by the proper SII) shows that single-cue mPL "
        "systematically double-counts shared leakage and therefore overstates per-cue "
        "attribution; Shapley redistributes the shared mass so that per-cue contributions sum "
        "exactly to <i>v</i>(<i>N</i>). In the 19 backup pairs the single-cue value errs "
        "oppositely and understates the contribution. De-duplication removed 44 overlap pairs "
        "but did not change this conclusion &mdash; 29 of the 44 were pure geometric duplicates, "
        "so the remaining overlap is genuine evidential redundancy between distinct regions "
        "(&sect;6.1). <b>Extremes.</b> Lake Bled is the strongest backup pair &mdash; church on "
        "an island &times; mountainous forested background, <i>I</i><super>SII</super> = "
        "+0.571, each cue nearly useless alone &mdash; followed by Inca stonework &times; "
        "vegetation at Cusco (+0.376). The strongest overlap pairs set a built landmark against "
        "the vegetation around it: Cuba, stone fortress walls &times; lush green landscape "
        "(&minus;0.407), and New York, historic building architecture &times; trees with autumn "
        "foliage (&minus;0.360). Only 2 of the 223 players have &phi; &lt; &minus;0.01."))
    S.extend(figure(
        "fig3_overview.png",
        "<b>Figure 6:</b> (a) every one of the 223 de-duplicated players, raw single-cue mPL "
        "against Shapley &phi;, sized by mask area and coloured by whether the removal resolves "
        "above that cue&rsquo;s own gray control floor; (b) per-category baseline mPL, "
        "Shapley-corrected &phi;, and the gray-arm resolvable share.",
        max_h=165))
    S.append(P(
        "<b>Two images end to end.</b> Figures 7 and 8 show the correction in both directions on "
        "single photographs. In the New York scene four cues sum to 1.231 in raw single-cue mPL "
        "against <i>v</i>(<i>N</i>) = 0.823 &mdash; the historic building architecture and the "
        "autumn foliage overlap heavily (<i>I</i><super>SII</super> = &minus;0.360) and both are "
        "marked down. In the Lake Bled scene the two cues sum to 0.775 raw against "
        "<i>v</i>(<i>N</i>) = 1.346 &mdash; the church and the mountain backdrop back each other "
        "up (+0.571), so both are marked <i>up</i>; a defence that removed either one alone "
        "would leave the location essentially as identifiable as before."))
    S.extend(figure(
        "fig7_case_newyork.png",
        "<b>Figure 7: Overlap.</b> New York, 4 cues. &Sigma;<i>v</i>({<i>k</i>}) = 1.231 &ne; "
        "<i>v</i>(<i>N</i>) = 0.823; Shapley redistributes the shared mass and the sum becomes "
        "exact.", max_h=172))
    S.extend(figure(
        "fig8_case_slovenia.png",
        "<b>Figure 8: Backup.</b> Lake Bled, Slovenia, 2 cues. &Sigma;<i>v</i>({<i>k</i>}) = "
        "0.775 &ne; <i>v</i>(<i>N</i>) = 1.346; both cues are credited above their standalone "
        "readings.", max_h=134))
    S.append(P(
        "<b>Robustness of the attribution operator.</b> Replacing the Shapley value by the "
        "Banzhaf value (uniform coalition weighting instead of permutation weighting) leaves the "
        "picture unchanged: Spearman &rho; = 0.984 across all cues, and the within-image rank "
        "correlation has median 1.000. An additive surrogate fitted by least squares to "
        "<i>v</i>(<i>S</i>) over all non-empty subsets explains a median <i>R</i><super>2</super> "
        "of only 0.668 per image, i.e. the non-additivity that motivates &sect;3 is not a "
        "rounding effect. The Harsanyi decomposition puts a median 65.5&#8201;% of the "
        "|dividend| mass at first order after de-duplication, 29.7&#8201;% at second, "
        "11.6&#8201;% at third and 3.2&#8201;% at fourth &mdash; pairwise interaction is the "
        "dominant non-additive term, which is what makes the order-2 estimator of &sect;4.1 "
        "work. All three of these share the same set function <i>v</i>(<i>S</i>) and therefore "
        "test the attribution operator, not the measurement; operator dependence of the "
        "measurement itself is what &sect;7 tests, and inventory dependence is what &sect;6 "
        "tests."))
    S.append(P(
        "<b>Reproducibility.</b> The 100-image sweep, the full subset lattice, the equal-area "
        "controls and the inpainting arm all run locally on the GeoRanker checkpoint with no API "
        "calls; only the cue-naming stage uses GPT-4o, and its output is cached per image. Full "
        "posteriors are persisted for every scored variant, so the set function, the Shapley "
        "values, the interaction indices and the de-duplication can all be recomputed from the "
        "stored JSON without re-running the adversary. The prior gray-fill numbers are retained "
        "in full as the robustness variant of the inpainting pipeline."))

    # -------------------------------------------------------- references ----
    S.append(H1("References"))
    for ref in [
        "[1]&nbsp; L. S. Shapley. <i>A Value for n-Person Games</i>. 1953.",
        "[2]&nbsp; P. Jia, S. Park, S. Gao, X. Zhao, Y. Li. <i>GeoRanker: Distance-Aware "
        "Ranking for Worldwide Image Geolocalization</i>. NeurIPS 2025.",
        "[3]&nbsp; Meta AI. <i>SAM 3: Segment Anything with Concepts</i>. 2025.",
        "[4]&nbsp; Chen et al. <i>Metric-normalized posterior leakage (mPL)</i>. 2026. (Full "
        "citation as in the companion proposal.)",
        "[5]&nbsp; R. Suvorov et al. <i>Resolution-robust Large Mask Inpainting with Fourier "
        "Convolutions (LaMa)</i>. WACV 2022.",
        "[6]&nbsp; M. Grabisch, M. Roubens. <i>An axiomatic approach to the concept of "
        "interaction among players in cooperative games</i>. Int. J. Game Theory, 1999.",
    ]:
        S.append(Paragraph(ref, STY["ref"]))
    return S


# ------------------------------------------------------------------ build ---

def page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVuSerif", 8.2)
    canvas.setFillColor(colors.HexColor("#555555"))
    if doc.page > 1:
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 8,
                          "From mPL to Shapley Attribution — GeoBayes technical note")
    canvas.drawCentredString(PAGE_W / 2, MARGIN - 12, str(doc.page))
    canvas.restoreState()


def main():
    register_fonts()
    STY.update(build_styles())
    if not os.path.isdir(ASSETS) or not os.path.exists(
            os.path.join(ASSETS, "fig1_pipeline.png")):
        print("missing paper/assets - run:  python paper/paper_figs.py")
        return 1
    EQ.update(render_equations())

    doc = BaseDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title="From mPL to Shapley Attribution: Quantifying Per-Cue "
              "Location-Privacy Leakage in Images",
        author="Xinpeng Xie", subject="GeoBayes technical note",
    )
    frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=page_furniture)])
    doc.build(build_story())
    print("wrote", OUT_PDF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
