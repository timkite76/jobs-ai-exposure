"""
AI Exposure of the US Job Market — Streamlit App

Analyzing how susceptible every occupation in the US economy is to AI and
automation, using data from the Bureau of Labor Statistics Occupational Outlook
Handbook. Original project by Andrej Karpathy: https://github.com/mariodian/jobs
"""

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Exposure of the US Job Market",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ────────────────────────────────────────────────────────────

DATA_PATH = pathlib.Path(__file__).parent / "data.json"


@st.cache_data
def load_data() -> pd.DataFrame:
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    # Clean up category names for display
    df["category_display"] = (
        df["category"]
        .str.replace("-", " ")
        .str.replace("and ", "& ")
        .str.title()
    )
    # Exposure tier labels
    tier_map = {
        0: "Minimal (0-1)", 1: "Minimal (0-1)",
        2: "Low (2-3)", 3: "Low (2-3)",
        4: "Moderate (4-5)", 5: "Moderate (4-5)",
        6: "High (6-7)", 7: "High (6-7)",
        8: "Very High (8-10)", 9: "Very High (8-10)", 10: "Very High (8-10)",
    }
    df["tier"] = df["exposure"].map(tier_map)
    # Wages at risk per occupation
    df["wages_at_risk"] = df["jobs"].fillna(0) * df["pay"].fillna(0)
    # Education years estimate for ROI analysis
    edu_years = {
        "No formal educational credential": 0,
        "High school diploma or equivalent": 0,
        "Postsecondary nondegree award": 1,
        "Some college, no degree": 2,
        "Associate's degree": 2,
        "Bachelor's degree": 4,
        "Master's degree": 6,
        "Doctoral or professional degree": 8,
    }
    df["edu_years"] = df["education"].map(edu_years)
    return df


df = load_data()

# ── Color scale (matches original green→red) ────────────────────────────


def exposure_color(score: float) -> str:
    """Return an rgb() string for an exposure score 0-10."""
    t = max(0.0, min(10.0, score)) / 10.0
    if t < 0.5:
        s = t / 0.5
        r = int(50 + s * 180)
        g = int(160 - s * 10)
        b = int(50 - s * 20)
    else:
        s = (t - 0.5) / 0.5
        r = int(230 + s * 25)
        g = int(150 - s * 110)
        b = int(30 - s * 10)
    return f"rgb({r},{g},{b})"


# Build a discrete colorscale for plotly (0-10 mapped to 0.0-1.0)
EXPOSURE_COLORSCALE = []
for i in range(11):
    EXPOSURE_COLORSCALE.append([i / 10, exposure_color(i)])

# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("AI Exposure of the US Job Market")
    st.caption(
        "342 occupations · Data from [BLS Occupational Outlook Handbook]"
        "(https://www.bls.gov/ooh/), scored by Gemini Flash · "
        "[GitHub](https://github.com/mariodian/jobs)"
    )

    valid = df.dropna(subset=["exposure", "jobs"])
    total_jobs = int(valid["jobs"].sum())
    weighted_avg = (valid["exposure"] * valid["jobs"]).sum() / valid["jobs"].sum()

    st.metric("Total Jobs", f"{total_jobs / 1e6:.0f}M")
    st.metric("Weighted Avg. Exposure", f"{weighted_avg:.1f} / 10")

    # Tier breakdown
    st.subheader("Jobs by Exposure Tier")
    tier_order = [
        "Minimal (0-1)", "Low (2-3)", "Moderate (4-5)",
        "High (6-7)", "Very High (8-10)",
    ]
    tier_colors = {
        "Minimal (0-1)": exposure_color(0.5),
        "Low (2-3)": exposure_color(2.5),
        "Moderate (4-5)": exposure_color(4.5),
        "High (6-7)": exposure_color(6.5),
        "Very High (8-10)": exposure_color(9),
    }
    tier_stats = (
        valid.groupby("tier")["jobs"]
        .sum()
        .reindex(tier_order)
        .fillna(0)
        .astype(int)
    )
    for tier_name, tier_jobs in tier_stats.items():
        pct = tier_jobs / total_jobs * 100
        col1, col2 = st.columns([3, 1])
        col1.markdown(
            f'<span style="color:{tier_colors[tier_name]}">●</span> {tier_name}',
            unsafe_allow_html=True,
        )
        col2.write(f"{tier_jobs / 1e6:.1f}M ({pct:.0f}%)")

    # Wages exposed
    high_exposure = valid[valid["exposure"] >= 7].dropna(subset=["pay"])
    wages_exposed = (high_exposure["jobs"] * high_exposure["pay"]).sum()
    st.metric(
        "Annual Wages in High-Exposure Jobs (7+)",
        f"${wages_exposed / 1e12:.1f}T",
    )

    st.divider()

    # Exposure by pay band
    st.subheader("Avg. Exposure by Pay Band")
    pay_bands = [
        ("<$35K", 0, 35_000),
        ("$35-50K", 35_000, 50_000),
        ("$50-75K", 50_000, 75_000),
        ("$75-100K", 75_000, 100_000),
        ("$100K+", 100_000, float("inf")),
    ]
    for label, lo, hi in pay_bands:
        band = valid[(valid["pay"] >= lo) & (valid["pay"] < hi)].dropna(subset=["pay"])
        if len(band) > 0:
            avg = (band["exposure"] * band["jobs"]).sum() / band["jobs"].sum()
        else:
            avg = 0
        st.markdown(
            f'<span style="color:{exposure_color(avg)}">●</span> '
            f"**{label}**: {avg:.1f}",
            unsafe_allow_html=True,
        )

    # Exposure by education
    st.subheader("Avg. Exposure by Education")
    edu_groups = [
        ("No degree/HS", ["No formal educational credential", "High school diploma or equivalent"]),
        ("Postsec/Assoc", ["Postsecondary nondegree award", "Some college, no degree", "Associate's degree"]),
        ("Bachelor's", ["Bachelor's degree"]),
        ("Master's", ["Master's degree"]),
        ("Doctoral/Prof", ["Doctoral or professional degree"]),
    ]
    for label, matches in edu_groups:
        grp = valid[valid["education"].isin(matches)]
        if len(grp) > 0:
            avg = (grp["exposure"] * grp["jobs"]).sum() / grp["jobs"].sum()
        else:
            avg = 0
        st.markdown(
            f'<span style="color:{exposure_color(avg)}">●</span> '
            f"**{label}**: {avg:.1f}",
            unsafe_allow_html=True,
        )

# ── Main area ────────────────────────────────────────────────────────────

view = st.radio(
    "View",
    ["Treemap", "Exposure vs Outlook", "Wages at Risk",
     "Education ROI", "Growth Paradox", "Category Heatmap",
     "Pay vs Safety", "Exposure Clusters", "Data Table"],
    horizontal=True,
    label_visibility="collapsed",
)

# Filters
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    categories = sorted(df["category_display"].dropna().unique())
    selected_cats = st.multiselect(
        "Filter by category", categories, default=[], placeholder="All categories"
    )
with col_f2:
    exposure_range = st.slider("Exposure range", 0, 10, (0, 10))
with col_f3:
    search = st.text_input("Search occupations", placeholder="e.g. software, nurse...")

# Apply filters
filtered = df.dropna(subset=["exposure", "jobs"]).copy()
if selected_cats:
    filtered = filtered[filtered["category_display"].isin(selected_cats)]
filtered = filtered[
    (filtered["exposure"] >= exposure_range[0])
    & (filtered["exposure"] <= exposure_range[1])
]
if search:
    filtered = filtered[
        filtered["title"].str.contains(search, case=False, na=False)
    ]

st.caption(f"Showing {len(filtered)} occupations · {int(filtered['jobs'].sum()):,} jobs")

# ── Treemap View ─────────────────────────────────────────────────────────

if view == "Treemap":
    treemap_df = filtered.copy()
    treemap_df["hover_text"] = treemap_df.apply(
        lambda r: (
            f"<b>{r['title']}</b><br>"
            f"AI Exposure: {r['exposure']}/10<br>"
            f"Median Pay: ${r['pay']:,.0f}<br>" if pd.notna(r["pay"]) else f"<b>{r['title']}</b><br>"
            f"Jobs (2024): {r['jobs']:,.0f}<br>"
            f"Outlook: {r['outlook']}% ({r['outlook_desc']})<br>" if pd.notna(r["outlook"]) else ""
            f"Education: {r['education']}<br>"
            f"<br><i>{r['exposure_rationale']}</i>"
        ),
        axis=1,
    )

    fig = px.treemap(
        treemap_df,
        path=[px.Constant("All Jobs"), "category_display", "title"],
        values="jobs",
        color="exposure",
        color_continuous_scale=EXPOSURE_COLORSCALE,
        range_color=[0, 10],
        custom_data=["exposure", "pay", "jobs", "outlook", "outlook_desc", "education", "exposure_rationale"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "AI Exposure: %{customdata[0]}/10<br>"
            "Median Pay: $%{customdata[1]:,.0f}<br>"
            "Jobs: %{customdata[2]:,.0f}<br>"
            "Outlook: %{customdata[3]}%% (%{customdata[4]})<br>"
            "Education: %{customdata[5]}<br>"
            "<extra></extra>"
        ),
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:,.0f} jobs",
    )

    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        height=700,
        coloraxis_colorbar=dict(
            title="AI Exposure",
            tickvals=[0, 2, 4, 6, 8, 10],
            ticktext=["0 (Safe)", "2", "4", "6", "8", "10 (Max)"],
        ),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Wages at Risk ────────────────────────────────────────────────────────

elif view == "Wages at Risk":
    st.subheader("Estimated Annual Wages at Risk by AI Exposure")
    st.caption(
        "Jobs x Median Pay = total wage dollars flowing through each occupation. "
        "Grouped by exposure tier to show where the economic impact concentrates."
    )

    war_df = filtered.dropna(subset=["pay"]).copy()
    war_df["wage_bill"] = war_df["jobs"] * war_df["pay"]

    # By tier
    tier_order_war = ["Minimal (0-1)", "Low (2-3)", "Moderate (4-5)", "High (6-7)", "Very High (8-10)"]
    tier_wages = war_df.groupby("tier")["wage_bill"].sum().reindex(tier_order_war, fill_value=0).reset_index()
    tier_wages.columns = ["Tier", "Total Wages"]
    tier_wages["color"] = [exposure_color(x) for x in [0.5, 2.5, 4.5, 6.5, 9.0]]

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        fig_tw = go.Figure(go.Bar(
            x=tier_wages["Tier"], y=tier_wages["Total Wages"],
            marker_color=tier_wages["color"],
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_tw.update_layout(
            height=400, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
            font=dict(color="#e0e0e8"),
            yaxis=dict(title="Annual Wages ($)", gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_tw, use_container_width=True)

    with col_w2:
        # By category
        cat_wages = (
            war_df.groupby("category_display")
            .agg(wage_bill=("wage_bill", "sum"), avg_exp=("exposure", "mean"))
            .sort_values("wage_bill", ascending=True)
            .tail(15)
        )
        fig_cw = go.Figure(go.Bar(
            y=cat_wages.index, x=cat_wages["wage_bill"], orientation="h",
            marker_color=cat_wages["avg_exp"].apply(exposure_color),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
        ))
        fig_cw.update_layout(
            title="Top 15 Categories by Wage Bill", height=400,
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
            font=dict(color="#e0e0e8", size=10),
            xaxis=dict(title="Annual Wages ($)", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=30, l=180, b=40),
        )
        st.plotly_chart(fig_cw, use_container_width=True)

    # Top individual occupations by wages at risk (exposure >= 7)
    st.subheader("Largest Wage Concentrations in High-Exposure Jobs (7+)")
    top_war = war_df[war_df["exposure"] >= 7].nlargest(20, "wage_bill")
    fig_top = go.Figure(go.Bar(
        y=top_war["title"].values[::-1], x=top_war["wage_bill"].values[::-1],
        orientation="h",
        marker_color=[exposure_color(e) for e in top_war["exposure"].values[::-1]],
        customdata=np.column_stack([top_war["exposure"].values[::-1], top_war["pay"].values[::-1], top_war["jobs"].values[::-1]]),
        hovertemplate="<b>%{y}</b><br>Wages: $%{x:,.0f}<br>Exposure: %{customdata[0]}/10<br>Pay: $%{customdata[1]:,.0f}<br>Jobs: %{customdata[2]:,.0f}<extra></extra>",
    ))
    fig_top.update_layout(
        height=500, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8", size=10),
        xaxis=dict(title="Annual Wage Bill ($)", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, l=250, b=40),
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ── Scatter: Exposure vs Outlook ─────────────────────────────────────────

elif view == "Exposure vs Outlook":
    scatter_df = filtered.dropna(subset=["outlook"]).copy()
    scatter_df["jobs_scaled"] = scatter_df["jobs"].clip(lower=100)

    fig = px.scatter(
        scatter_df,
        x="exposure",
        y="outlook",
        size="jobs_scaled",
        color="exposure",
        color_continuous_scale=EXPOSURE_COLORSCALE,
        range_color=[0, 10],
        hover_name="title",
        hover_data={
            "exposure": True,
            "outlook": True,
            "pay": ":$,.0f",
            "jobs": ":,.0f",
            "education": True,
            "jobs_scaled": False,
        },
        labels={
            "exposure": "AI Exposure (0-10)",
            "outlook": "Employment Outlook (%)",
            "pay": "Median Pay",
            "jobs": "Jobs (2024)",
        },
        size_max=50,
    )

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.add_vline(x=5, line_dash="dot", line_color="rgba(255,255,255,0.2)")

    # Quadrant annotations
    fig.add_annotation(x=2, y=20, text="Low Exposure<br>Growing", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.3)", size=12))
    fig.add_annotation(x=8, y=20, text="High Exposure<br>Growing", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.3)", size=12))
    fig.add_annotation(x=2, y=-10, text="Low Exposure<br>Declining", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.3)", size=12))
    fig.add_annotation(x=8, y=-10, text="High Exposure<br>Declining", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.3)", size=12))

    fig.update_layout(
        height=700,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            range=[-0.5, 10.5],
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
        ),
        coloraxis_colorbar=dict(
            title="AI Exposure",
            tickvals=[0, 2, 4, 6, 8, 10],
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Education ROI ────────────────────────────────────────────────────────

elif view == "Education ROI":
    st.subheader("Education ROI Under AI Pressure")
    st.caption(
        "Are expensive degrees leading people into the most exposed careers? "
        "Each bubble is one occupation. X = education investment, Y = AI exposure, size = employment."
    )

    roi_df = filtered.dropna(subset=["edu_years", "pay"]).copy()
    roi_df["jobs_scaled"] = roi_df["jobs"].clip(lower=100)

    fig_roi = px.scatter(
        roi_df, x="edu_years", y="exposure", size="jobs_scaled",
        color="pay", color_continuous_scale="Viridis",
        hover_name="title",
        hover_data={"edu_years": False, "exposure": True, "pay": ":$,.0f",
                    "jobs": ":,.0f", "education": True, "jobs_scaled": False},
        labels={"edu_years": "Education (years beyond HS)", "exposure": "AI Exposure (0-10)",
                "pay": "Median Pay"},
        size_max=40,
    )
    fig_roi.update_layout(
        height=600, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", dtick=1,
                   ticktext=["HS/None", "1yr cert", "Associate's", "", "Bachelor's", "", "Master's", "", "Doctoral"],
                   tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Summary table: avg exposure and pay by education level
    st.subheader("Average Exposure & Pay by Education Level")
    edu_summary = (
        roi_df.groupby("education")
        .apply(lambda g: pd.Series({
            "Avg Exposure": np.average(g["exposure"], weights=g["jobs"]),
            "Avg Pay": np.average(g["pay"], weights=g["jobs"]),
            "Total Jobs": g["jobs"].sum(),
        }))
        .sort_values("Avg Exposure", ascending=False)
    )
    edu_summary["Avg Pay"] = edu_summary["Avg Pay"].apply(lambda x: f"${x:,.0f}")
    edu_summary["Total Jobs"] = edu_summary["Total Jobs"].apply(lambda x: f"{x:,.0f}")
    edu_summary["Avg Exposure"] = edu_summary["Avg Exposure"].apply(lambda x: f"{x:.1f}")
    st.dataframe(edu_summary, use_container_width=True)

# ── Growth Paradox ───────────────────────────────────────────────────────

elif view == "Growth Paradox":
    st.subheader("Growth Paradox: High Exposure + Strong BLS Growth")
    st.caption(
        "Occupations where BLS projects strong growth (5%+) but AI exposure is high (7+). "
        "Either BLS underestimates AI disruption, or these roles will transform rather than disappear."
    )

    paradox_df = filtered.dropna(subset=["outlook"]).copy()
    paradox_high = paradox_df[(paradox_df["exposure"] >= 7) & (paradox_df["outlook"] >= 5)]
    paradox_high = paradox_high.sort_values("outlook", ascending=False)

    if len(paradox_high) == 0:
        st.info("No occupations match the current filters for this view.")
    else:
        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(
            x=paradox_df["exposure"], y=paradox_df["outlook"],
            mode="markers",
            marker=dict(size=6, color="rgba(255,255,255,0.15)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig_par.add_trace(go.Scatter(
            x=paradox_high["exposure"], y=paradox_high["outlook"],
            mode="markers+text",
            marker=dict(
                size=paradox_high["jobs"].clip(lower=1000).apply(lambda x: max(8, min(40, x / 100000))),
                color=[exposure_color(e) for e in paradox_high["exposure"]],
                line=dict(width=1, color="white"),
            ),
            text=paradox_high["title"].apply(lambda t: t[:25] + "..." if len(t) > 25 else t),
            textposition="top center", textfont=dict(size=9, color="#e0e0e8"),
            customdata=np.column_stack([paradox_high["pay"].fillna(0), paradox_high["jobs"], paradox_high["outlook"]]),
            hovertemplate="<b>%{text}</b><br>Exposure: %{x}/10<br>Growth: %{customdata[2]}%<br>Pay: $%{customdata[0]:,.0f}<br>Jobs: %{customdata[1]:,.0f}<extra></extra>",
            showlegend=False,
        ))
        # Highlight zone
        fig_par.add_shape(type="rect", x0=7, x1=10.5, y0=5, y1=paradox_df["outlook"].max() * 1.1,
                          fillcolor="rgba(255,100,100,0.05)", line=dict(color="rgba(255,100,100,0.3)", dash="dot"))
        fig_par.add_annotation(x=8.5, y=paradox_df["outlook"].max() * 1.05,
                               text="PARADOX ZONE", showarrow=False,
                               font=dict(color="rgba(255,100,100,0.6)", size=14, family="monospace"))
        fig_par.update_layout(
            height=600, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
            font=dict(color="#e0e0e8"),
            xaxis=dict(title="AI Exposure", gridcolor="rgba(255,255,255,0.06)", range=[0, 10.5]),
            yaxis=dict(title="BLS Projected Growth (%)", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_par, use_container_width=True)

        st.subheader(f"Paradox Occupations ({len(paradox_high)} found)")
        para_table = paradox_high[["title", "exposure", "outlook", "outlook_desc", "pay", "jobs", "education"]].copy()
        para_table.columns = ["Occupation", "Exposure", "Growth %", "Outlook", "Pay", "Jobs", "Education"]
        para_table = para_table.sort_values("Growth %", ascending=False)
        st.dataframe(para_table, use_container_width=True, column_config={
            "Pay": st.column_config.NumberColumn(format="$%d"),
            "Jobs": st.column_config.NumberColumn(format="%d"),
        })

# ── Category Heatmap ─────────────────────────────────────────────────────

elif view == "Category Heatmap":
    st.subheader("Category vs. Exposure Tier Heatmap")
    st.caption("Cell color = total employment. Reveals which entire sectors are concentrated at high exposure.")

    heat_df = filtered.copy()
    tier_order_h = ["Minimal (0-1)", "Low (2-3)", "Moderate (4-5)", "High (6-7)", "Very High (8-10)"]
    pivot = heat_df.pivot_table(index="category_display", columns="tier", values="jobs",
                                aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=tier_order_h, fill_value=0)
    # Sort categories by total jobs descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="YlOrRd", hovertemplate="<b>%{y}</b><br>%{x}<br>Jobs: %{z:,.0f}<extra></extra>",
        colorbar=dict(title="Jobs"),
    ))
    fig_heat.update_layout(
        height=max(500, len(pivot) * 24), paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, l=200),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Weighted avg exposure per category
    st.subheader("Weighted Average Exposure by Category")
    cat_exp = (
        heat_df.groupby("category_display")
        .apply(lambda g: pd.Series({
            "Weighted Exposure": np.average(g["exposure"], weights=g["jobs"]) if g["jobs"].sum() > 0 else 0,
            "Total Jobs": g["jobs"].sum(),
        }))
        .sort_values("Weighted Exposure", ascending=True)
    )
    fig_ce = go.Figure(go.Bar(
        y=cat_exp.index, x=cat_exp["Weighted Exposure"], orientation="h",
        marker_color=[exposure_color(e) for e in cat_exp["Weighted Exposure"]],
        customdata=cat_exp["Total Jobs"].values,
        hovertemplate="<b>%{y}</b><br>Avg Exposure: %{x:.1f}<br>Jobs: %{customdata:,.0f}<extra></extra>",
    ))
    fig_ce.update_layout(
        height=max(400, len(cat_exp) * 22), paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8", size=10),
        xaxis=dict(title="Weighted Avg Exposure", gridcolor="rgba(255,255,255,0.06)", range=[0, 10]),
        margin=dict(t=10, l=200, b=40),
    )
    st.plotly_chart(fig_ce, use_container_width=True)

# ── Pay vs Safety ────────────────────────────────────────────────────────

elif view == "Pay vs Safety":
    st.subheader("The Safety Penalty: Do Low-Exposure Jobs Pay Less?")
    st.caption("Pay distribution for each exposure tier. Is there a wage premium for AI-exposed work?")

    pay_df = filtered.dropna(subset=["pay"]).copy()

    fig_box = px.box(
        pay_df, x="tier", y="pay", color="tier",
        category_orders={"tier": ["Minimal (0-1)", "Low (2-3)", "Moderate (4-5)", "High (6-7)", "Very High (8-10)"]},
        color_discrete_map={
            "Minimal (0-1)": exposure_color(0.5), "Low (2-3)": exposure_color(2.5),
            "Moderate (4-5)": exposure_color(4.5), "High (6-7)": exposure_color(6.5),
            "Very High (8-10)": exposure_color(9),
        },
        hover_data={"title": True, "pay": ":$,.0f", "jobs": ":,.0f", "tier": False},
        labels={"pay": "Median Annual Pay ($)", "tier": "Exposure Tier"},
        points="all",
    )
    fig_box.update_layout(
        height=500, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"), showlegend=False,
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Weighted median pay by tier
    st.subheader("Weighted Average Pay by Tier")
    tier_pay = (
        pay_df.groupby("tier")
        .apply(lambda g: pd.Series({
            "Weighted Avg Pay": np.average(g["pay"], weights=g["jobs"]),
            "Total Jobs": g["jobs"].sum(),
            "Occupations": len(g),
        }))
        .reindex(["Minimal (0-1)", "Low (2-3)", "Moderate (4-5)", "High (6-7)", "Very High (8-10)"])
    )
    tier_pay["Weighted Avg Pay"] = tier_pay["Weighted Avg Pay"].apply(lambda x: f"${x:,.0f}")
    tier_pay["Total Jobs"] = tier_pay["Total Jobs"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(tier_pay, use_container_width=True)

    # Scatter: pay vs exposure with trend
    st.subheader("Pay vs. Exposure (every occupation)")
    fig_pv = px.scatter(
        pay_df, x="exposure", y="pay", size=pay_df["jobs"].clip(lower=100),
        color="exposure", color_continuous_scale=EXPOSURE_COLORSCALE, range_color=[0, 10],
        hover_name="title", hover_data={"exposure": True, "pay": ":$,.0f", "jobs": ":,.0f"},
        labels={"exposure": "AI Exposure", "pay": "Median Pay ($)"},
        size_max=35, trendline="ols",
    )
    fig_pv.update_layout(
        height=450, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10),
    )
    st.plotly_chart(fig_pv, use_container_width=True)

# ── Exposure Clusters ────────────────────────────────────────────────────

elif view == "Exposure Clusters":
    st.subheader("Occupation Clusters by Exposure Rationale")
    st.caption(
        "TF-IDF on the LLM rationale text + KMeans clustering reveals *why* occupations are exposed — "
        "grouping by shared themes (digital output, data processing, physical barriers, etc.) rather than BLS category."
    )

    cluster_df = filtered.dropna(subset=["exposure_rationale"]).copy()
    n_clusters = st.slider("Number of clusters", 3, 12, 6)

    @st.cache_data
    def compute_clusters(rationales, n_clust):
        tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        X = tfidf.fit_transform(rationales)
        km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.toarray())
        # Top terms per cluster
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf.get_feature_names_out()
        top_terms = {}
        for i in range(n_clust):
            top_terms[i] = ", ".join(terms[ind] for ind in order_centroids[i, :5])
        return labels, coords, top_terms

    labels, coords, top_terms = compute_clusters(cluster_df["exposure_rationale"].tolist(), n_clusters)
    cluster_df["cluster"] = labels
    cluster_df["x"] = coords[:, 0]
    cluster_df["y"] = coords[:, 1]
    cluster_df["cluster_label"] = cluster_df["cluster"].map(
        {i: f"C{i}: {top_terms[i]}" for i in range(n_clusters)}
    )

    fig_cl = px.scatter(
        cluster_df, x="x", y="y", color="cluster_label",
        hover_name="title",
        hover_data={"exposure": True, "x": False, "y": False, "cluster_label": False,
                    "pay": ":$,.0f", "jobs": ":,.0f"},
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "cluster_label": "Cluster"},
        size=cluster_df["jobs"].clip(lower=100),
        size_max=30,
    )
    fig_cl.update_layout(
        height=600, paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
        legend=dict(font=dict(size=10)),
        margin=dict(t=10),
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    # Cluster summary table
    st.subheader("Cluster Profiles")
    for i in sorted(cluster_df["cluster"].unique()):
        c = cluster_df[cluster_df["cluster"] == i]
        avg_exp = np.average(c["exposure"], weights=c["jobs"]) if c["jobs"].sum() > 0 else 0
        with st.expander(f"Cluster {i}: {top_terms[i]}  |  {len(c)} occupations  |  Avg exposure: {avg_exp:.1f}"):
            st.write(f"**Top keywords:** {top_terms[i]}")
            st.write(f"**Total jobs:** {c['jobs'].sum():,.0f}")
            examples = c.nlargest(5, "jobs")[["title", "exposure", "pay", "jobs"]].copy()
            examples.columns = ["Occupation", "Exposure", "Pay", "Jobs"]
            st.dataframe(examples, use_container_width=True, column_config={
                "Pay": st.column_config.NumberColumn(format="$%d"),
                "Jobs": st.column_config.NumberColumn(format="%d"),
            })

# ── Data Table View ──────────────────────────────────────────────────────

elif view == "Data Table":
    table_df = filtered[
        ["title", "category_display", "exposure", "pay", "jobs", "outlook",
         "outlook_desc", "education", "exposure_rationale"]
    ].copy()
    table_df.columns = [
        "Occupation", "Category", "AI Exposure", "Median Pay", "Jobs (2024)",
        "Outlook %", "Outlook", "Education", "AI Rationale",
    ]
    table_df = table_df.sort_values("AI Exposure", ascending=False)

    st.dataframe(
        table_df,
        use_container_width=True,
        height=650,
        column_config={
            "AI Exposure": st.column_config.ProgressColumn(
                min_value=0, max_value=10, format="%d/10",
            ),
            "Median Pay": st.column_config.NumberColumn(format="$%d"),
            "Jobs (2024)": st.column_config.NumberColumn(format="%d"),
            "Outlook %": st.column_config.NumberColumn(format="%d%%"),
        },
    )

# ── Histogram below main chart ──────────────────────────────────────────

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Employment Distribution by Exposure Score")
    hist_data = (
        filtered.groupby("exposure")["jobs"]
        .sum()
        .reindex(range(11), fill_value=0)
        .reset_index()
    )
    hist_data.columns = ["Exposure Score", "Total Jobs"]
    hist_data["color"] = hist_data["Exposure Score"].apply(exposure_color)

    fig_hist = go.Figure(
        go.Bar(
            x=hist_data["Exposure Score"],
            y=hist_data["Total Jobs"],
            marker_color=hist_data["color"],
            hovertemplate="Score %{x}: %{y:,.0f} jobs<extra></extra>",
        )
    )
    fig_hist.update_layout(
        height=300,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8"),
        xaxis=dict(
            title="AI Exposure Score",
            tickvals=list(range(11)),
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="Total Jobs",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Top 15 Most Exposed Occupations (by employment)")
    top_exposed = (
        filtered[filtered["exposure"] >= 7]
        .nlargest(15, "jobs")
        .sort_values("jobs")
    )
    fig_bar = go.Figure(
        go.Bar(
            y=top_exposed["title"],
            x=top_exposed["jobs"],
            orientation="h",
            marker_color=top_exposed["exposure"].apply(exposure_color),
            customdata=top_exposed[["exposure", "pay"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Jobs: %{x:,.0f}<br>"
                "Exposure: %{customdata[0]}/10<br>"
                "Pay: $%{customdata[1]:,.0f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig_bar.update_layout(
        height=300,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#e0e0e8", size=10),
        xaxis=dict(
            title="Jobs (2024)",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, l=200, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Methodology ──────────────────────────────────────────────────────────

st.divider()

with st.expander("Methodology: Scoring Algorithm & Heuristics", expanded=False):
    st.markdown("""
### Data Pipeline

This analysis follows a five-stage pipeline built on top of the **Bureau of Labor Statistics
[Occupational Outlook Handbook](https://www.bls.gov/ooh/)** (OOH), which covers **342 occupations**
spanning every sector of the US economy.

| Stage | Description |
|-------|-------------|
| **1. Scrape** | Playwright downloads the raw HTML for all 342 BLS occupation pages (job duties, work environment, education, pay, projections). |
| **2. Parse** | BeautifulSoup converts raw HTML into clean Markdown, preserving the structured detail of each occupation profile. |
| **3. Tabulate** | Structured fields are extracted into a flat table: median pay, entry education, job count, growth outlook, and SOC code. |
| **4. Score** | Each occupation's full Markdown description is sent to an LLM (Gemini Flash via OpenRouter) with a calibrated scoring rubric. The model returns a 0-10 exposure score and a written rationale. |
| **5. Merge** | CSV statistics and AI exposure scores are combined into a single dataset for visualization. |

---

### AI Exposure Scoring Rubric

Each occupation is scored on a single **AI Exposure** axis from **0 to 10**, measuring *how much
AI will reshape that occupation*. The score captures two dimensions:

- **Direct automation** — AI performing tasks currently done by humans (e.g., code generation,
  document drafting, data entry)
- **Indirect productivity effects** — AI making each worker so productive that fewer are needed
  (e.g., one analyst doing the work of five with AI tools)

#### Primary Heuristic: Digital vs. Physical Work

The single strongest signal in the rubric is whether the job's **work product is fundamentally
digital**:

- If the job can be done entirely from a home office on a computer — writing, coding, analyzing,
  communicating — AI exposure is **inherently high (7+)**, because AI capabilities in digital
  domains are advancing rapidly. Even if today's AI can't handle every aspect, the trajectory
  is steep and the ceiling is very high.
- Conversely, jobs requiring **physical presence, manual skill, or real-time human interaction**
  in the physical world have a natural barrier to AI exposure.

#### Calibration Anchors

The LLM is given explicit calibration examples to ensure consistent scoring across all 342 occupations:

| Score | Tier | Heuristic | Examples |
|-------|------|-----------|----------|
| **0-1** | Minimal | Almost entirely physical, hands-on work in unpredictable environments. AI has essentially no impact on daily work. | Roofers, landscapers, commercial divers |
| **2-3** | Low | Mostly physical or interpersonal. AI helps with peripheral tasks (scheduling, paperwork) but doesn't touch the core job. | Electricians, plumbers, firefighters, dental hygienists |
| **4-5** | Moderate | A mix of physical/interpersonal and knowledge work. AI meaningfully assists the information-processing parts, but a substantial share still requires human presence. | Registered nurses, police officers, veterinarians |
| **6-7** | High | Predominantly knowledge work with some need for human judgment, relationships, or physical presence. AI tools are already useful and workers using AI may be substantially more productive. | Teachers, managers, accountants, journalists |
| **8-9** | Very High | Almost entirely done on a computer. All core tasks — writing, coding, analyzing, designing — are in domains where AI is rapidly improving. The occupation faces major restructuring. | Software developers, graphic designers, translators, paralegals |
| **10** | Maximum | Routine information processing, fully digital, no physical component. AI can already do most of it today. | Data entry clerks, telemarketers |

#### LLM Scoring Process

- **Model:** Google Gemini Flash (via OpenRouter API)
- **Temperature:** 0.2 (low variance for consistent, deterministic scoring)
- **Input:** The full Markdown profile of each occupation (duties, environment, education, pay, projections)
- **Output:** A structured JSON with an integer `exposure` score (0-10) and a 2-3 sentence `rationale`
- **Incremental checkpointing:** Results are saved after each occupation so the pipeline can resume if interrupted

---

### Visualization Algorithms

**Treemap (Squarified Layout)**

The treemap uses the **squarified treemap algorithm** (Bruls, Huizing & van Wijk, 2000), which
optimizes rectangle aspect ratios to be as close to 1:1 as possible for readability:

- Occupations are first grouped by BLS category
- Category blocks are laid out proportional to their total employment
- Within each category, individual occupations are laid out proportional to their job count
- **Area** = number of jobs (2024 employment figures)
- **Color** = AI exposure score on a green (safe) to red (exposed) continuous scale

**Scatter Plot (Exposure vs. Outlook)**

Maps each occupation on two axes to reveal which jobs face a "double threat":

- **X-axis:** AI Exposure score (0-10)
- **Y-axis:** BLS employment outlook (projected % change 2024-2034)
- **Bubble size:** proportional to current employment
- **Quadrant interpretation:** Upper-left = safe & growing; Lower-right = exposed & declining

**Weighted Statistics**

All aggregate statistics (average exposure, tier breakdowns, pay/education comparisons) are
**job-weighted** — each occupation contributes proportionally to its employment count, so that
an occupation with 4M workers has 1000x the influence of one with 4K workers. This prevents
the 342 occupations from being treated as equally important when they vary by orders of magnitude
in employment.

---

### Limitations

- Scores reflect a single LLM's assessment at a point in time; different models or prompts may yield different scores
- The BLS OOH groups some related occupations together, so granularity varies
- Exposure scores measure *potential for AI impact*, not a timeline — a score of 8 doesn't mean 80% of jobs disappear tomorrow
- The scoring rubric intentionally weights digital/physical nature heavily, which may underweight other factors like regulatory barriers or union protections
""")

