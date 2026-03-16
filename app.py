"""
AI Exposure of the US Job Market — Streamlit App

Analyzing how susceptible every occupation in the US economy is to AI and
automation, using data from the Bureau of Labor Statistics Occupational Outlook
Handbook. Original project by Andrej Karpathy: https://github.com/mariodian/jobs
"""

import json
import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    ["Treemap", "Exposure vs Outlook", "Data Table"],
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

