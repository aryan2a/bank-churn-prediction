"""Look and feel of the Streamlit app: colours, CSS, and the few custom pieces (risk strip, chart styling).

Colour is data here. Coral means "pushes towards leaving" and blue means "pushes towards staying"; nothing else on the
page uses those two colours. Everything else is black, white and grey. Both colours were checked against a pure black
surface for lightness, colour-blind separation and contrast.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go

# ---- tokens -------------------------------------------------------------------------------------------------------
BG, PANEL, LINE = "#000000", "#0E0E0F", "#26262A"
INK, INK2, MUTED = "#F3F3F1", "#A4A4AA", "#8A8A91"
HEAT, COOL, NEUTRAL = "#E8673C", "#4C95D6", "#5A5A62"
BAND_RAMP = ["#2A2A2F", "#4A3028", "#7A3F27", "#B0532E", HEAT]          # lowest -> highest risk band
BANDS = ["Lowest", "Low", "Medium", "High", "Highest"]
DISPLAY, BODY = '"Bricolage Grotesque", "Source Sans", system-ui, sans-serif', '"Instrument Sans", "Source Sans", system-ui, sans-serif'

_FONTS = [("Bricolage Grotesque", 600, "bricolage-grotesque-latin-600-normal"), ("Bricolage Grotesque", 700, "bricolage-grotesque-latin-700-normal"),
          ("Instrument Sans", 400, "instrument-sans-latin-400-normal"), ("Instrument Sans", 500, "instrument-sans-latin-500-normal"),
          ("Instrument Sans", 600, "instrument-sans-latin-600-normal")]
_FONT_CSS = "".join(
    f'@font-face{{font-family:"{fam}";font-weight:{w};font-style:normal;font-display:swap;'
    f'src:url(app/static/fonts/{f}.woff2) format("woff2")}}' for fam, w, f in _FONTS)

CSS = f"""<style>
{_FONT_CSS}
.stApp {{ background:{BG}; font-family:{BODY}; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {{ font-family:{BODY}; }}
.block-container {{ padding-top:2.6rem; padding-bottom:4rem; max-width:1180px; }}

/* headings and big numbers use the display face */
h1, h2, h3, h4, [data-testid="stMetricValue"], .v-title {{ font-family:{DISPLAY} !important; letter-spacing:-0.02em; }}
h1 {{ font-size:2.7rem !important; font-weight:700 !important; line-height:1.05 !important; padding:0 0 .35rem 0 !important; }}
h3 {{ font-size:1.35rem !important; font-weight:600 !important; }}
h4 {{ font-size:1.05rem !important; font-weight:600 !important; color:{INK}; }}
.lede {{ color:{INK2}; font-size:1.08rem; line-height:1.55; max-width:62ch; margin:.1rem 0 1.6rem 0; }}
.lede b {{ color:{INK}; font-weight:500; }}

/* tabs: plain text with one underline */
.stTabs [data-baseweb="tab-list"] {{ gap:2rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ height:auto; padding:.7rem 0; background:transparent; color:{MUTED}; font-size:1rem; }}
.stTabs [aria-selected="true"] {{ color:{INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{INK}; height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* numbers: a hairline and a big figure, no boxes */
[data-testid="stMetric"] {{ border-top:1px solid {LINE}; padding:.85rem 0 0 0; }}
[data-testid="stMetricLabel"] p {{ color:{INK2}; font-size:.9rem; }}
[data-testid="stMetricValue"] {{ font-size:2.35rem !important; font-weight:600 !important; line-height:1.15; }}
[data-testid="stMetricDelta"] {{ color:{INK2} !important; }}

/* the customer form is the only bordered panel */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; }}
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="base-input"] {{ background:#151517 !important; border-color:{LINE} !important; }}
.grp {{ color:{INK2}; font-size:.85rem; font-weight:500; margin:.9rem 0 .15rem 0; }}
.grp:first-child {{ margin-top:.1rem; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ white-space:nowrap; background:transparent; border:1px solid #3A3A40; color:{INK}; border-radius:999px; padding:.35rem 1rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{INK}; color:{INK}; background:transparent; }}

/* verdict */
.verdict {{ display:flex; align-items:flex-start; gap:.9rem; margin:.1rem 0 1rem 0; }}
.verdict .dot {{ width:14px; height:14px; border-radius:50%; flex:none; margin:.7rem 0 0 .35rem; }}
.verdict.flag .dot {{ background:{HEAT}; box-shadow:0 0 0 5px rgba(232,103,60,.18); }}
.verdict.ok .dot {{ background:{COOL}; box-shadow:0 0 0 5px rgba(76,149,214,.18); }}
.v-title {{ font-size:2rem; font-weight:700; line-height:1.1; }}
.verdict.flag .v-title {{ color:{HEAT}; }} .verdict.ok .v-title {{ color:{COOL}; }}
.v-sub {{ color:{INK2}; margin-top:.2rem; }}

/* risk strip */
.rs {{ margin:.2rem 0 1.4rem 0; }}
.rs-track {{ position:relative; display:flex; gap:3px; height:14px; margin:34px 0 8px 0; }}
.rs-seg {{ flex:1; border-radius:3px; }}
.rs-mark {{ position:absolute; top:-12px; bottom:-12px; width:4px; margin-left:-2px; background:{INK}; border-radius:2px; box-shadow:0 0 0 2px {BG}; }}
.rs-line {{ position:absolute; top:-22px; bottom:-6px; border-left:2px dashed {INK2}; margin-left:-1px; }}
.rs-line span {{ position:absolute; top:-4px; white-space:nowrap; font-size:.8rem; color:{INK2}; }}
.rs-line span.r {{ left:8px; }} .rs-line span.l {{ right:8px; }}
.rs-labels {{ display:flex; gap:3px; }}
.rs-labels span {{ flex:1; font-size:.8rem; color:{MUTED}; }}
.rs-labels span.on {{ color:{INK}; font-weight:600; }}
.rs-note {{ color:{MUTED}; font-size:.82rem; margin-top:.55rem; }}

/* small status lines in the sidebar */
.status {{ display:flex; align-items:center; gap:.5rem; font-size:.95rem; }}
.status .dot {{ width:9px; height:9px; border-radius:50%; }}
.status.ok .dot {{ background:{COOL}; }} .status.bad .dot {{ background:{HEAT}; }}
section[data-testid="stSidebar"] {{ background:#080809; border-right:1px solid {LINE}; }}

/* tables, alerts, captions, inline code */
.stApp code {{ color:{INK} !important; background:#17171A !important; border:1px solid {LINE}; border-radius:4px; padding:.05rem .3rem; }}
[data-testid="stCaptionContainer"] {{ color:{MUTED}; }}
[data-testid="stAlert"] {{ border-radius:10px; }}
</style>"""


def risk_strip(percentile: float, threshold_percentile: float | None, threshold: float, band: str) -> str:
    """Customers ranked low to high by risk score, in five equal groups (the risk bands).
    The white marker is where this customer sits; the dashed line is where calls start."""
    pct = max(0.0, min(100.0, float(percentile)))
    segs = "".join(f'<span class="rs-seg" style="background:{c}"></span>' for c in BAND_RAMP)
    labels = "".join(f'<span class="{"on" if b == band else ""}">{b}</span>' for b in BANDS)
    line = ""
    if threshold_percentile is not None:
        tp = max(0.0, min(100.0, float(threshold_percentile)))
        side = "l" if tp >= 78 else "r"
        line = f'<div class="rs-line" style="left:{tp}%"><span class="{side}">calls start at {threshold:.2f}</span></div>'
    aria = f"This customer is riskier than {pct:.0f}% of customers and is in the {band} band."
    note = "Customers ranked by risk score, in five equal groups. The white marker is this customer; the dashed line is where calls start."
    return (f'<div class="rs" role="img" aria-label="{html.escape(aria)}"><div class="rs-track">{segs}{line}'
            f'<div class="rs-mark" style="left:{pct}%"></div></div><div class="rs-labels">{labels}</div>'
            f'<div class="rs-note">{note}</div></div>')


def verdict(flagged: bool, score: float, threshold: float) -> str:
    if flagged:
        return (f'<div class="verdict flag"><span class="dot"></span><div><div class="v-title">Call this customer</div>'
                f'<div class="v-sub">Risk score {score:.2f} is at or above the call threshold of {threshold:.2f}.</div></div></div>')
    return (f'<div class="verdict ok"><span class="dot"></span><div><div class="v-title">No call needed</div>'
            f'<div class="v-sub">Risk score {score:.2f} is below the call threshold of {threshold:.2f}.</div></div></div>')


def status(ok: bool, text: str) -> str:
    return f'<div class="status {"ok" if ok else "bad"}"><span class="dot"></span>{html.escape(text)}</div>'


# ---- charts -------------------------------------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int, **layout) -> go.Figure:
    layout.setdefault("margin", dict(l=10, r=10, t=10, b=10))
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, Source Sans, sans-serif", color=INK2, size=13),
        hoverlabel=dict(bgcolor="#1B1B1E", bordercolor=LINE, font=dict(color=INK, family="Instrument Sans, Source Sans, sans-serif")),
        **layout)
    fig.update_xaxes(gridcolor="#18181B", zerolinecolor="#3A3A40", linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor="#18181B", zerolinecolor="#3A3A40", linecolor=LINE, tickfont=dict(color=INK2), title_font=dict(color=INK2))
    return fig


def drivers_chart(drivers: list[dict]) -> go.Figure:
    """Horizontal bars growing left (lowers risk, blue) or right (raises risk, coral) from a centre line."""
    d = drivers[::-1]
    xs = [x["impact"] for x in d]
    fig = go.Figure(go.Bar(
        x=xs, y=[f'{x["label"]}: {x["value"]}' for x in d], orientation="h",
        marker=dict(color=[HEAT if v > 0 else COOL for v in xs]),
        text=[f"{v:+.2f}" for v in xs], textposition="outside", cliponaxis=False, textfont=dict(color=INK2, size=12),
        hovertemplate="%{y}<br>pushes the score %{x:+.2f}<extra></extra>"))
    lim = max(abs(v) for v in xs) * 1.35 if xs else 1
    style_fig(fig, 80 + 38 * len(d), margin=dict(l=10, r=24, t=34, b=10), bargap=0.45,
              xaxis=dict(range=[-lim, lim], showticklabels=False, showgrid=False, zeroline=True, fixedrange=True),
              yaxis=dict(automargin=True, fixedrange=True, showgrid=False, linecolor="rgba(0,0,0,0)"))
    fig.add_annotation(x=0, xref="x", y=1, yref="paper", yshift=18, xanchor="right", xshift=-8, showarrow=False,
                       text="lowers risk", font=dict(color=COOL, size=12))
    fig.add_annotation(x=0, xref="x", y=1, yref="paper", yshift=18, xanchor="left", xshift=8, showarrow=False,
                       text="raises risk", font=dict(color=HEAT, size=12))
    return fig


def score_histogram(scores, threshold: float) -> go.Figure:
    import numpy as np
    edges = np.linspace(0, 1, 41)
    counts, _ = np.histogram(scores, bins=edges)
    mid = (edges[:-1] + edges[1:]) / 2
    flagged = edges[:-1] >= threshold - 1e-9
    fig = go.Figure(go.Bar(x=mid, y=counts, width=0.0235, marker=dict(color=[HEAT if f else NEUTRAL for f in flagged]),
                           hovertemplate="score %{x:.2f}<br>%{y:,} customers<extra></extra>"))
    style_fig(fig, 240, margin=dict(l=10, r=10, t=26, b=30), bargap=0.08,
              xaxis=dict(title="Risk score", range=[0, 1], fixedrange=True, showgrid=False), yaxis=dict(title="Customers", fixedrange=True, automargin=True))
    fig.add_vline(x=threshold, line_color=INK2, line_dash="dash", line_width=1.5)
    fig.add_annotation(x=threshold, xref="x", y=1, yref="paper", yshift=14, xanchor="left", xshift=6, showarrow=False,
                       text=f"calls start at {threshold:.2f}", font=dict(color=INK2, size=12))
    return fig
