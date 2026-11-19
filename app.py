"""Exoplanet Explorer - interactive dashboard (Streamlit + Plotly), styled like a modern React admin dashboard.

Run the pipeline first (python exo_pipeline.py), then:  python -m streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"

st.set_page_config(page_title="Exoplanet Explorer", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design tokens
BG, SURFACE, BORDER = "#0B1120", "#111A2E", "#1F2A44"
TEXT, MUTED, ACCENT = "#E2E8F0", "#8A97B1", "#6366F1"
GRID = "#1B2640"
PALETTE = ["#6366F1", "#22D3EE", "#F59E0B", "#34D399", "#F472B6", "#A78BFA", "#FB7185", "#60A5FA"]
SIZE_ORDER = ["Earth-size", "Super-Earth", "Neptune-size", "Jupiter-size"]
SIZE_COLORS = {"Earth-size": "#22D3EE", "Super-Earth": "#F59E0B", "Neptune-size": "#6366F1", "Jupiter-size": "#F472B6"}

pio.templates.default = "plotly_dark"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = "Viridis"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #0B1120; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.4rem 2rem 3rem; max-width: 1600px; }

/* sidebar */
[data-testid="stSidebar"] { background: #0E1628; border-right: 1px solid #1F2A44; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
.sb-brand { display:flex; align-items:center; gap:10px; margin-bottom: 4px; }
.sb-logo { width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#6366F1,#22D3EE);
           display:flex; align-items:center; justify-content:center; font-size:18px; }
.sb-name { font-weight:700; font-size:1.02rem; color:#E2E8F0; line-height:1.1; }
.sb-sub { font-size:.74rem; color:#8A97B1; }
.sb-section { font-size:.78rem; font-weight:600; color:#8A97B1; margin: 18px 0 2px; letter-spacing:.02em; }
[data-testid="stSidebar"] label p { font-size:.84rem; color:#CBD5E1; font-weight:500; }
[data-testid="stSidebar"] hr { border-color:#1F2A44; }

/* top bar */
.topbar { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
          padding: 4px 0 18px; }
.topbar h1 { margin:0; font-size:1.6rem; font-weight:700; color:#F1F5F9; letter-spacing:-.02em; padding:0; }
.topbar p { margin:4px 0 0; color:#8A97B1; font-size:.92rem; }
.pills { display:flex; gap:8px; flex-wrap:wrap; }
.pill { font-size:.76rem; color:#CBD5E1; background:#111A2E; border:1px solid #1F2A44; padding:5px 11px;
        border-radius:999px; }
.pill b { color:#F1F5F9; font-weight:600; }

/* KPI cards (st.metric) */
[data-testid="stMetric"] { background:#111A2E; border:1px solid #1F2A44; border-radius:12px; padding:16px 18px;
                           box-shadow: 0 1px 0 rgba(255,255,255,.03) inset; }
[data-testid="stMetricLabel"] p { color:#8A97B1; font-size:.8rem; font-weight:500; }
[data-testid="stMetricValue"] { color:#F1F5F9; font-weight:700; font-size:1.65rem; letter-spacing:-.02em; }
[data-testid="stMetricDelta"] { color:#8A97B1 !important; font-size:.76rem; }
[data-testid="stMetricDelta"] svg { display:none; }

/* tabs as a segmented control */
[data-baseweb="tab-list"] { gap:4px; background:#0F172A; border:1px solid #1F2A44; border-radius:10px;
                            padding:4px; width:fit-content; max-width:100%; overflow-x:auto; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }
button[role="tab"] { height:34px; padding:0 14px; border-radius:7px; color:#8A97B1; font-weight:500; font-size:.86rem; }
button[role="tab"]:hover { color:#E2E8F0; background:#162038; }
button[role="tab"][aria-selected="true"] { background:#6366F1; color:#fff; }
button[role="tab"][aria-selected="true"] p { color:#fff; }
[data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* chart cards */
[class*="st-key-card_"] { background:#111A2E; border:1px solid #1F2A44; border-radius:14px; padding:18px 20px 12px; }
[class*="st-key-card_"] [data-testid="stMetric"] { background:transparent; border:none; padding:0; box-shadow:none; }
.card-t { font-weight:600; color:#F1F5F9; font-size:.98rem; }
.card-s { color:#8A97B1; font-size:.8rem; margin-top:2px; }
.card-h { margin-bottom:6px; }

/* callouts and chips */
.callout { border-left:3px solid #6366F1; background:#131D36; border-radius:0 10px 10px 0; padding:12px 16px;
           color:#CBD5E1; font-size:.88rem; line-height:1.55; margin-top:6px; }
.chip { display:inline-block; font-size:.76rem; color:#CBD5E1; background:#18233E; border:1px solid #243255;
        padding:3px 10px; border-radius:999px; margin:0 6px 6px 0; }
.chip.good { color:#6EE7B7; background:#0F2A27; border-color:#14503F; }
.note { color:#8A97B1; font-size:.8rem; margin-top:4px; }

/* inputs / tables / buttons */
[data-testid="stDataFrame"] { border:1px solid #1F2A44; border-radius:10px; overflow:hidden; }
.stDownloadButton button { background:#6366F1; color:#fff; border:none; border-radius:8px; font-weight:600; }
.stDownloadButton button:hover { background:#4F52D9; color:#fff; }
[data-baseweb="select"] > div { background:#0F172A; border-color:#1F2A44; }
h2, h3 { color:#F1F5F9; letter-spacing:-.01em; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

_card_n = [0]


def card(where, title, sub=None):
    """A bordered, titled card (like a React <Card>). Returns the container to draw into."""
    _card_n[0] += 1
    box = where.container(key=f"card_{_card_n[0]}")
    head = f'<div class="card-h"><div class="card-t">{title}</div>'
    head += f'<div class="card-s">{sub}</div>' if sub else ""
    box.markdown(head + "</div>", unsafe_allow_html=True)
    return box


def plot(fig, box, height=340, legend_top=True):
    fig.update_layout(
        height=height, title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12), colorway=PALETTE,
        margin=dict(l=4, r=4, t=8, b=4),
        hoverlabel=dict(bgcolor="#0B1120", bordercolor=BORDER, font=dict(family="Inter, sans-serif", color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="", font=dict(size=11)) if legend_top
        else dict(title_text=""),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, ticks="outside", tickcolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    box.plotly_chart(fig, width="stretch")


@st.cache_data
def load():
    df = pd.read_csv(PROC / "exoplanets_clean.csv")
    solar = pd.read_csv(PROC / "solar_system.csv")
    wlog = json.loads((PROC / "wrangling_log.json").read_text())
    df["size_class"] = pd.Categorical(df["size_class"], SIZE_ORDER, ordered=True)
    return df, solar, wlog


if not (PROC / "exoplanets_clean.csv").exists():
    st.error("Processed data not found. Run `python exo_pipeline.py` first.")
    st.stop()

df, solar, wlog = load()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.markdown(
    '<div class="sb-brand"><div class="sb-logo">🪐</div><div><div class="sb-name">Exoplanet Explorer</div>'
    '<div class="sb-sub">Filters apply to every chart</div></div></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sb-section">Discovery</div>', unsafe_allow_html=True)
y0, y1 = int(df["disc_year"].min()), int(df["disc_year"].max())
years = st.sidebar.slider("Discovery year", y0, y1, (1995, y1))
methods = st.sidebar.multiselect("Discovery method", sorted(df["method"].unique()), default=sorted(df["method"].unique()))
st.sidebar.markdown('<div class="sb-section">Planet</div>', unsafe_allow_html=True)
sizes = st.sidebar.multiselect("Planet size class", SIZE_ORDER, default=SIZE_ORDER)
max_ly = st.sidebar.slider("Max distance (light-years)", 10, 30000, 30000, step=10)
st.sidebar.markdown('<div class="sb-section">Quick filters</div>', unsafe_allow_html=True)
only_hz = st.sidebar.checkbox("Only habitable-zone planets")
only_matched = st.sidebar.checkbox("Only planets with star-catalogue data")
st.sidebar.divider()
st.sidebar.caption("Data: Open Exoplanet Catalogue + HYG star database, wrangled with pandas.")

f = df[df["disc_year"].between(*years) & df["method"].isin(methods) & df["size_class"].isin(sizes)]
f = f[(f["distance_ly"].isna() & (max_ly >= 30000)) | (f["distance_ly"] <= max_ly)]
if only_hz:
    f = f[f["in_hz"]]
if only_matched:
    f = f[f["star_matched"]]

# ------------------------------------------------------------------ header + KPIs
st.markdown(
    f'<div class="topbar"><div><h1>Exoplanet Explorer</h1>'
    f'<p>Planets outside our Solar System: when we found them, what they are like, and where life might exist.</p></div>'
    f'<div class="pills"><span class="pill"><b>{len(df):,}</b> confirmed planets</span>'
    f'<span class="pill"><b>{y0}&ndash;{y1}</b> discovery years</span></div></div>', unsafe_allow_html=True)

if f.empty:
    st.warning("No planets match these filters. Widen the filters in the sidebar.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Planets", f"{len(f):,}", f"{len(f) / len(df):.0%} of catalogue", delta_color="off", delta_arrow="off")
c2.metric("Host stars", f"{f['host'].nunique():,}")
c3.metric("Median distance", f"{f['distance_ly'].median():,.0f} ly" if f["distance_ly"].notna().any() else "n/a")
c4.metric("Habitable-zone", f"{int(f['in_hz'].sum())}", f"{int(f['hz_candidate'].sum())} small/rocky",
          delta_color="off", delta_arrow="off")
c5.metric("Multi-planet systems", f"{f.loc[f['n_planets_in_system'] > 1, 'host'].nunique():,}")

st.write("")
tab = st.tabs(["Overview", "Population", "Host stars", "Sky map", "Habitable zone", "Data quality", "Data"])

# ------------------------------------------------------------------ overview
with tab[0]:
    a, b = st.columns([2, 1])
    per_year = f.groupby(["disc_year", "method"]).size().reset_index(name="planets")
    fig = px.bar(per_year, x="disc_year", y="planets", color="method",
                 labels={"disc_year": "year", "planets": "planets discovered"})
    plot(fig, card(a, "Discoveries per year", "Planets found each year, split by detection method"), 380)
    m = f["method"].value_counts().reset_index()
    m.columns = ["method", "planets"]
    fig = px.pie(m, names="method", values="planets", hole=0.62)
    fig.update_traces(marker_line_color="#111A2E", marker_line_width=2, textinfo="percent")
    plot(fig, card(b, "Share by method", "Which technique found them"), 380, legend_top=False)

    a, b = st.columns([2, 1])
    sc = f["size_class"].value_counts().reindex(SIZE_ORDER).reset_index()
    sc.columns = ["size_class", "planets"]
    fig = px.bar(sc, x="size_class", y="planets", color="size_class", color_discrete_map=SIZE_COLORS,
                 labels={"size_class": "", "planets": "planets"})
    fig.update_layout(showlegend=False)
    plot(fig, card(a, "Planets by size class", "Earth-size to Jupiter-size"), 300)
    with card(b, "How to read this"):
        st.markdown(
            '<div class="callout">Most planets were found by the transit method with space telescopes such as Kepler. '
            'Short-period giants are easiest to detect, so the catalogue is biased toward them.</div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ population
with tab[1]:
    box = card(st, "Radius vs orbital period", "Each dot is a planet; log scales on both axes")
    show_solar = box.checkbox("Overlay our Solar System", value=True)
    p = f.dropna(subset=["radius_earth", "period_days"])
    fig = px.scatter(p, x="period_days", y="radius_earth", color="size_class", color_discrete_map=SIZE_COLORS,
                     hover_name="name", hover_data=["host", "method", "disc_year", "distance_ly"],
                     log_x=True, log_y=True, opacity=0.65,
                     labels={"period_days": "orbital period (days)", "radius_earth": "radius (Earth = 1)"})
    if show_solar:
        fig.add_trace(go.Scatter(x=solar["period_days"], y=solar["radius_earth"], mode="markers+text", text=solar["name"],
                                 textposition="top center", textfont=dict(color=TEXT, size=11),
                                 marker=dict(symbol="star", size=14, color="white"), name="Solar System"))
    plot(fig, box, 480)
    mm = f.dropna(subset=["mass_earth", "radius_earth"])
    if len(mm):
        fig = px.scatter(mm, x="mass_earth", y="radius_earth", color="size_class", color_discrete_map=SIZE_COLORS,
                         log_x=True, log_y=True, hover_name="name", opacity=0.65,
                         labels={"mass_earth": "mass (Earth = 1)", "radius_earth": "radius (Earth = 1)"})
        box2 = card(st, "Mass vs radius", f"{len(mm):,} planets with both measured")
        plot(fig, box2, 420)
    st.markdown('<div class="note">Rocky planets follow a steep mass-radius relation; above roughly 100 Earth masses, '
                'adding mass no longer adds size.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ host stars
with tab[2]:
    s = f.dropna(subset=["star_temp_k", "star_lum_lsun"])
    a, b = st.columns([3, 2])
    fig = px.scatter(s, x="star_temp_k", y="star_lum_lsun", color="star_class", log_y=True, hover_name="host",
                     opacity=0.65, category_orders={"star_class": list("OBAFGKM")},
                     labels={"star_temp_k": "temperature (K)", "star_lum_lsun": "luminosity (Sun = 1)"})
    fig.update_xaxes(autorange="reversed")
    plot(fig, card(a, "HR diagram", "Host star temperature vs luminosity"), 400)
    sc_ = f["star_class"].value_counts().reindex(list("OBAFGKM")).dropna().reset_index()
    sc_.columns = ["star_class", "planets"]
    fig = px.bar(sc_, x="star_class", y="planets", labels={"star_class": "star class"})
    plot(fig, card(b, "Planets by host star class", "O is hottest, M is coolest"), 400)
    met = f.dropna(subset=["star_metallicity"])
    if len(met):
        fig = px.histogram(met, x="star_metallicity", color="size_class", color_discrete_map=SIZE_COLORS, nbins=40,
                           barmode="overlay", opacity=0.65, labels={"star_metallicity": "host star metallicity"})
        plot(fig, card(st, "Host star metallicity by planet size", "Metal-rich stars tend to host more giant planets"), 320)
    st.markdown('<div class="note">Our Sun is a G-class star. Most hosts are Sun-like (F, G, K) because surveys targeted them.</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ sky map
with tab[3]:
    k = f.dropna(subset=["ra_deg", "dec_deg"]).copy()
    k["lon"] = k["ra_deg"] - 180
    k["distance_ly"] = k["distance_ly"].fillna(k["distance_ly"].median())
    fig = px.scatter_geo(k, lat="dec_deg", lon="lon", color=np.log10(k["distance_ly"]), hover_name="name",
                         hover_data={"lon": False, "dec_deg": ":.1f", "ra_deg": ":.1f", "distance_ly": ":,.0f"},
                         projection="mollweide", color_continuous_scale="Turbo", opacity=0.75)
    fig.update_layout(coloraxis_colorbar=dict(title="log10 ly", thickness=10, len=0.7))
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=False, showcoastlines=False, showframe=True, framecolor=BORDER,
                    lonaxis=dict(showgrid=True, gridcolor=GRID), lataxis=dict(showgrid=True, gridcolor=GRID))
    plot(fig, card(st, "Sky map", "Host star position (RA / Dec). Colour shows log10 distance in light-years."), 520,
         legend_top=False)
    st.markdown('<div class="note">The dense patch is the field watched by the Kepler telescope in Cygnus and Lyra.</div>',
                unsafe_allow_html=True)
    near = f.nsmallest(10, "distance_ly")[["name", "distance_ly", "size_class", "method", "disc_year"]]
    box = card(st, "Ten nearest planets")
    box.dataframe(near.round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ habitable zone
with tab[4]:
    st.markdown('<div class="callout">A planet is flagged when its orbit lies where a star\'s light gives roughly Earth-like '
                'warmth (0.53 to 1.1 times Earth\'s sunlight). This is a simplified estimate, not proof of habitability.</div>',
                unsafe_allow_html=True)
    st.write("")
    hz = f[f["in_hz"]]
    if hz.empty:
        st.info("No habitable-zone planets in the current selection.")
    else:
        fig = px.scatter(hz, x="sma_au", y="radius_earth", color="star_class", size="n_planets_in_system",
                         hover_name="name", hover_data=["host", "distance_ly", "star_temp_k"], log_x=True,
                         labels={"sma_au": "orbit distance (AU)", "radius_earth": "radius (Earth = 1)"})
        fig.add_hline(y=2, line_dash="dot", line_color=MUTED, annotation_text="rocky-planet size limit (~2 Earth radii)",
                      annotation_font_color=MUTED)
        plot(fig, card(st, "Planets inside their star's habitable zone", "Bubble size shows planets in the system"), 400)
        cand = hz[hz["hz_candidate"]].sort_values("distance_ly")[
            ["name", "host", "radius_earth", "mass_earth", "sma_au", "star_class", "distance_ly", "disc_year"]]
        box = card(st, f"Small, possibly rocky candidates ({len(cand)})", "Sorted by distance from us")
        box.dataframe(cand.round(2), hide_index=True, width="stretch")

    box = card(st, "Planet profile", "Pick any planet in the current selection")
    pick = box.selectbox("Choose a planet", sorted(f["name"]), index=0)
    r = f[f["name"] == pick].iloc[0]
    fmt = lambda v, u="": "n/a" if pd.isna(v) else f"{v:,.2f}{u}"
    x1, x2, x3, x4 = box.columns(4)
    x1.metric("Radius (Earth = 1)", fmt(r["radius_earth"]))
    x2.metric("Mass (Earth = 1)", fmt(r["mass_earth"]))
    x3.metric("Orbit period", fmt(r["period_days"], " d"))
    x4.metric("Distance", fmt(r["distance_ly"], " ly"))
    chips = (f'<span class="chip">{r["size_class"]}</span><span class="chip">{r["method"]}</span>'
             f'<span class="chip">Discovered {int(r["disc_year"])}</span>'
             f'<span class="chip">Host: {r["host"]}</span>')
    if bool(r["in_hz"]):
        chips += '<span class="chip good">Inside habitable zone</span>'
    box.markdown(f'<div style="margin-top:10px">{chips}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ data quality
with tab[5]:
    q1, q2, q3 = st.columns(3)
    q1.metric("Raw rows", f"{wlog['raw_rows']:,}")
    q2.metric("Clean rows", f"{wlog['clean_rows']:,}", f"-{wlog['raw_rows'] - wlog['clean_rows']} removed",
              delta_color="off", delta_arrow="off")
    q3.metric("Columns (raw to clean)", f"{wlog['raw_cols']} to {wlog['clean_cols']}")
    st.write("")
    mb = pd.Series(wlog["missing_before"]).sort_values(ascending=False).head(12)
    ma = pd.Series(wlog["missing_after"]).reindex(mb.index)
    mdf = pd.DataFrame({"column": mb.index, "raw": mb.values, "after cleaning": ma.values}).melt(
        "column", var_name="stage", value_name="missing %")
    fig = px.bar(mdf.dropna(), x="missing %", y="column", color="stage", barmode="group", orientation="h")
    fig.update_yaxes(autorange="reversed")
    plot(fig, card(st, "Missing values by column",
                   "Columns dropped for sparsity show no 'after' bar"), 440)
    box = card(st, "What was done", "The wrangling steps behind this dashboard")
    for step in wlog["steps"]:
        box.markdown(step)

# ------------------------------------------------------------------ data
with tab[6]:
    cols = ["name", "host", "method", "disc_year", "size_class", "radius_earth", "mass_earth", "period_days",
            "sma_au", "distance_ly", "star_class", "star_temp_k", "in_hz", "constellation"]
    box = card(st, "Planet table", f"{len(f):,} planets match the current filters")
    box.dataframe(f[cols], hide_index=True, width="stretch", height=500)
    box.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode(), "exoplanets_filtered.csv", "text/csv")