"""
Conway's Game of Life for Data Science Pattern Discovery
Complete Streamlit Application
Author: Senior Data Scientist / Python Engineer
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import io
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Conway's Game of Life – Data Science Edition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Base font */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    border: 1px solid rgba(100,100,255,0.2);
}
.main-header h1 {
    font-size: 2.4rem;
    color: #e0e0ff;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
}
.main-header p {
    color: #9090cc;
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(100,120,255,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #7070aa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a0a0ff;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #50d080;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #c0c0ff;
    border-left: 4px solid #6060ff;
    padding-left: 0.8rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Rule box */
.rule-box {
    background: rgba(40,40,80,0.6);
    border: 1px solid rgba(80,80,160,0.4);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #d0d0ff;
}

/* Hinglish card */
.hinglish-card {
    background: linear-gradient(135deg, #1a0a2e, #2e1a0e);
    border: 1px solid rgba(255,150,50,0.3);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
    color: #ffd0a0;
    font-size: 0.92rem;
}

/* Tag pills */
.pill {
    display: inline-block;
    background: rgba(80,80,200,0.3);
    border: 1px solid rgba(100,100,255,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    color: #b0b0ff;
    margin: 0.2rem;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #3030a0, #6030c0);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧬 Conway's Game of Life — Data Science Edition</h1>
  <p>Transform real-world datasets into living cellular automata · Discover hidden patterns · Explore emergent behaviour</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=600, seed=42):
    """Generate a synthetic socio-economic dataset with 500+ records."""
    rng = np.random.default_rng(seed)
    urban_factor   = rng.beta(2, 2, n)
    pop_density    = urban_factor * 900 + rng.normal(50, 20, n)
    income         = urban_factor * 70000 + rng.normal(5000, 8000, n)
    pollution      = urban_factor * 60 + rng.normal(10, 10, n) + rng.uniform(0, 20, n)
    education      = urban_factor * 80 + rng.normal(10, 12, n)
    employment     = 50 + urban_factor * 40 + rng.normal(0, 6, n)
    healthcare     = urban_factor * 70 + rng.normal(15, 15, n)
    internet       = urban_factor * 85 + rng.normal(5, 10, n)
    crime          = (1 - urban_factor) * 70 + rng.normal(10, 15, n) + pollution * 0.1

    df = pd.DataFrame({
        "Population_Density": np.clip(pop_density, 10, 1000).round(2),
        "Income":             np.clip(income, 5000, 150000).round(2),
        "Pollution_Index":    np.clip(pollution, 0, 100).round(2),
        "Education_Score":    np.clip(education, 0, 100).round(2),
        "Employment_Rate":    np.clip(employment, 10, 100).round(2),
        "Healthcare_Score":   np.clip(healthcare, 0, 100).round(2),
        "Internet_Penetration": np.clip(internet, 0, 100).round(2),
        "Crime_Rate":         np.clip(crime, 0, 100).round(2),
    })

    # Inject a few missing values
    for col in rng.choice(df.columns, 3, replace=False):
        df.loc[rng.choice(df.index, 5, replace=False), col] = np.nan

    return df


# ─────────────────────────────────────────────
# CONWAY ENGINE
# ─────────────────────────────────────────────
def data_to_grid(series: pd.Series, grid_size: int, threshold_mode: str,
                 percentile: float = 50.0, custom_val: float = None) -> np.ndarray:
    """Convert a numeric series to a 2-D binary grid using the chosen threshold."""
    vals = series.dropna().values
    if threshold_mode == "Mean":
        thr = np.mean(vals)
    elif threshold_mode == "Median":
        thr = np.median(vals)
    elif threshold_mode == "Percentile":
        thr = np.percentile(vals, percentile)
    else:  # Custom
        thr = custom_val if custom_val is not None else np.mean(vals)

    total_cells = grid_size * grid_size
    # Tile/repeat series to fill grid
    tiled = np.tile(vals, (total_cells // len(vals)) + 1)[:total_cells]
    grid = (tiled >= thr).astype(int).reshape(grid_size, grid_size)
    return grid, thr


def count_neighbors(grid: np.ndarray) -> np.ndarray:
    """Compute the number of alive neighbours for every cell (toroidal wrap)."""
    g = grid
    N = (np.roll(g, -1, axis=0) + np.roll(g, 1, axis=0) +
         np.roll(g, -1, axis=1) + np.roll(g, 1, axis=1) +
         np.roll(np.roll(g, -1, axis=0), -1, axis=1) +
         np.roll(np.roll(g, -1, axis=0),  1, axis=1) +
         np.roll(np.roll(g,  1, axis=0), -1, axis=1) +
         np.roll(np.roll(g,  1, axis=0),  1, axis=1))
    return N


def step(grid: np.ndarray) -> np.ndarray:
    """Apply Conway's four rules to produce the next generation grid."""
    n = count_neighbors(grid)
    alive = grid == 1
    # Rule 1 – Underpopulation:   alive & <2 neighbours → dies
    # Rule 2 – Survival:          alive & 2-3 neighbours → lives
    # Rule 3 – Overpopulation:    alive & >3 neighbours → dies
    # Rule 4 – Reproduction:      dead  &  3 neighbours → born
    new_grid = np.zeros_like(grid)
    new_grid[alive & ((n == 2) | (n == 3))] = 1
    new_grid[(~alive) & (n == 3)] = 1
    return new_grid


def run_simulation(initial_grid: np.ndarray, generations: int):
    """Run Conway simulation, return list of grids + statistics per generation."""
    grids = [initial_grid.copy()]
    stats = []
    grid = initial_grid.copy()

    for gen in range(1, generations + 1):
        prev_alive = grid.sum()
        grid = step(grid)
        alive = int(grid.sum())
        total = grid.size
        dead  = total - alive
        pct   = alive / total * 100
        growth = alive - prev_alive

        # Stability: check if identical to previous
        stable = int(np.array_equal(grid, grids[-1]))

        stats.append({
            "Generation":    gen,
            "Alive":         alive,
            "Dead":          dead,
            "Alive_Pct":     round(pct, 2),
            "Growth":        growth,
            "Stable":        stable,
        })
        grids.append(grid.copy())

        # Early exit on extinction
        if alive == 0:
            break

    return grids, pd.DataFrame(stats)


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "conway", ["#0f0c29", "#a0a0ff"], N=2)

def plot_grid(grid: np.ndarray, title: str, figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#0f0c29")
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, color="#c0c0ff", fontsize=11, pad=8)
    ax.axis("off")
    legend_elements = [
        Patch(facecolor="#a0a0ff", label="Alive (1)"),
        Patch(facecolor="#0f0c29", edgecolor="#444", label="Dead (0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.4, labelcolor="white", fontsize=8)
    fig.tight_layout()
    return fig


def plot_stat_chart(df_stats: pd.DataFrame, y_col: str, title: str, color: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#12122a")
    ax.plot(df_stats["Generation"], df_stats[y_col], color=color, linewidth=2)
    ax.fill_between(df_stats["Generation"], df_stats[y_col],
                    alpha=0.15, color=color)
    ax.set_title(title, color="#c0c0ff", fontsize=10)
    ax.set_xlabel("Generation", color="#8080aa", fontsize=8)
    ax.set_ylabel(y_col, color="#8080aa", fontsize=8)
    ax.tick_params(colors="#8080aa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")

    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)
    df_raw = generate_dataset(600, int(seed))

    feature = st.selectbox("Feature to Convert", df_raw.columns.tolist(),
                           index=0)

    threshold_mode = st.radio(
        "Threshold Strategy",
        ["Mean", "Median", "Percentile", "Custom"],
        index=0,
    )

    percentile_val = 50.0
    custom_thr = float(df_raw[feature].mean())
    if threshold_mode == "Percentile":
        percentile_val = st.slider("Percentile", 1.0, 99.0, 50.0, 1.0)
    elif threshold_mode == "Custom":
        mn = float(df_raw[feature].min())
        mx = float(df_raw[feature].max())
        custom_thr = st.slider("Custom Threshold", mn, mx,
                               float(df_raw[feature].mean()))

    grid_size = st.slider("Grid Size (NxN)", 10, 50, 25, 5)
    generations = st.slider("Generations", 10, 200, 50, 10)
    anim_speed = st.slider("Animation Speed (ms / frame)", 50, 500, 150, 50)

    st.markdown("---")
    run_btn   = st.button("▶  Run Simulation", use_container_width=True)
    reset_btn = st.button("↺  Reset", use_container_width=True)

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Conway's Game of Life applied to socio-economic data. "
               "Each cell represents whether a region's metric is above or below threshold.")

# ─────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────
if "sim_done" not in st.session_state or reset_btn:
    st.session_state.sim_done = False
    st.session_state.grids = []
    st.session_state.df_stats = pd.DataFrame()
    st.session_state.initial_grid = None
    st.session_state.threshold_used = None

df = df_raw.copy()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Dataset Explorer",
    "🔬 Conversion & Grid",
    "▶ Simulation",
    "📈 Analytics Dashboard",
    "🧠 Pattern Detection",
    "📚 Data Science Concepts",
    "🌍 Real-World Insights",
    "🤖 AI / ML Connection",
    "🇮🇳 Hinglish Explainer",
    "📄 Report",
])


# ══════════════════════════════════════════════
# TAB 1 – DATASET EXPLORER
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.info("📌 Synthetic socio-economic dataset with **600 records** across 8 urban/environmental features.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", len(df))
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    with st.expander("🔍 Raw Data Preview", expanded=True):
        st.dataframe(df.head(30), use_container_width=True)

    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df.describe().T.style.background_gradient(cmap="coolwarm"),
                     use_container_width=True)

    with st.expander("❓ Missing Value Analysis"):
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Feature", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(miss, use_container_width=True)
        st.caption("Missing values are excluded when computing thresholds.")

    with st.expander("🔗 Correlation Matrix"):
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        fig_corr.patch.set_facecolor("#0f0c29")
        ax_corr.set_facecolor("#0f0c29")
        corr = df.corr(numeric_only=True)
        im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_yticks(range(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right",
                                color="#c0c0ff", fontsize=8)
        ax_corr.set_yticklabels(corr.columns, color="#c0c0ff", fontsize=8)
        ax_corr.set_title("Feature Correlation Matrix", color="#c0c0ff")
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

    with st.expander("📊 Feature Distributions"):
        cols_per_row = 4
        feature_list = df.columns.tolist()
        for i in range(0, len(feature_list), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, feat in enumerate(feature_list[i:i+cols_per_row]):
                with row_cols[j]:
                    fig_h, ax_h = plt.subplots(figsize=(3, 2.2))
                    fig_h.patch.set_facecolor("#0f0c29")
                    ax_h.set_facecolor("#12122a")
                    ax_h.hist(df[feat].dropna(), bins=25, color="#6060ff",
                              edgecolor="#0f0c29", alpha=0.9)
                    ax_h.set_title(feat.replace("_", " "), color="#c0c0ff",
                                   fontsize=8)
                    ax_h.tick_params(colors="#8080aa", labelsize=6)
                    for sp in ax_h.spines.values():
                        sp.set_edgecolor("#333355")
                    fig_h.tight_layout()
                    st.pyplot(fig_h)


# ══════════════════════════════════════════════
# TAB 2 – CONVERSION & GRID
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Data → Binary Grid Conversion</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **How it works:**  
    Each value in the selected feature is compared against a *threshold*.  
    - Value **≥ threshold** → Cell **Alive** (1) 🟣  
    - Value **< threshold** → Cell **Dead** (0) ⚫
    """)

    # Compute grid
    initial_grid, thr = data_to_grid(
        df[feature], grid_size, threshold_mode, percentile_val, custom_thr)

    c1, c2, c3 = st.columns(3)
    c1.metric("Feature", feature.replace("_", " "))
    c2.metric("Threshold Value", f"{thr:.2f}")
    c3.metric("Alive Cells (Initial)",
              f"{int(initial_grid.sum())} / {grid_size*grid_size}")

    col_g, col_e = st.columns([1, 1])
    with col_g:
        st.markdown("**Initial Grid**")
        st.pyplot(plot_grid(initial_grid, f"Initial Grid – {feature.replace('_',' ')}",
                            figsize=(5, 5)))

    with col_e:
        st.markdown("**Threshold Strategy Explained**")
        threshold_explain = {
            "Mean":       "Cells are alive if the feature value is above the **arithmetic mean**. Balances high and low values but sensitive to outliers.",
            "Median":     "Cells are alive if the feature value is above the **median** (50th percentile). Robust to outliers — always gives ~50% alive cells.",
            "Percentile": f"Cells are alive if the value exceeds the **{percentile_val:.0f}th percentile**. Useful for focusing on extreme values.",
            "Custom":     f"Cells are alive if the value exceeds your chosen threshold of **{custom_thr:.2f}**. Full manual control.",
        }
        st.info(threshold_explain[threshold_mode])

        st.markdown("**Binary Classification View**")
        sample = df[feature].dropna().head(20).reset_index(drop=True)
        sample_df = pd.DataFrame({
            "Value": sample.values.round(2),
            "≥ Threshold": sample.values >= thr,
            "Cell State": ["🟣 Alive" if v >= thr else "⚫ Dead" for v in sample.values],
        })
        st.dataframe(sample_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**Conway's Rules (Visual Reference)**")
    r1, r2, r3, r4 = st.columns(4)
    rules = [
        ("😵 Underpopulation", "Alive cell with <2 neighbours → Dies. Loneliness kills."),
        ("😊 Survival", "Alive cell with 2-3 neighbours → Survives. Balanced community."),
        ("😵 Overpopulation", "Alive cell with >3 neighbours → Dies. Resource depletion."),
        ("🐣 Reproduction", "Dead cell with exactly 3 neighbours → Born. Critical mass reached."),
    ]
    for col, (title, desc) in zip([r1,r2,r3,r4], rules):
        col.markdown(f'<div class="rule-box"><strong>{title}</strong><br>{desc}</div>',
                     unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 – SIMULATION
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">▶ Conway Simulation</div>',
                unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Running simulation…"):
            ig, thr = data_to_grid(
                df[feature], grid_size, threshold_mode, percentile_val, custom_thr)
            grids, df_stats = run_simulation(ig, generations)
            st.session_state.grids = grids
            st.session_state.df_stats = df_stats
            st.session_state.initial_grid = ig
            st.session_state.threshold_used = thr
            st.session_state.sim_done = True

    if not st.session_state.sim_done:
        st.warning("👈 Configure settings in the sidebar and click **Run Simulation**.")
    else:
        grids     = st.session_state.grids
        df_stats  = st.session_state.df_stats
        ig        = st.session_state.initial_grid

        total_gens = len(grids) - 1  # grids[0] = initial
        st.success(f"✅ Simulation complete — {total_gens} generations evolved.")

        # 3-panel grid comparison
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_grid(grids[0], "Generation 0 (Initial)"))
        with c2:
            mid = total_gens // 2
            st.pyplot(plot_grid(grids[mid], f"Generation {mid} (Mid)"))
        with c3:
            st.pyplot(plot_grid(grids[-1], f"Generation {total_gens} (Final)"))

        # Animation
        st.markdown("**🎬 Generation-by-Generation Playback**")
        gen_slider = st.slider("Select Generation", 0, total_gens, 0)
        st.pyplot(plot_grid(grids[gen_slider], f"Generation {gen_slider}",
                            figsize=(6, 6)))

        row = df_stats[df_stats["Generation"] == gen_slider]
        if not row.empty:
            r = row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Generation", int(gen_slider))
            m2.metric("Alive",  int(r["Alive"]))
            m3.metric("Dead",   int(r["Dead"]))
            m4.metric("Alive %", f"{r['Alive_Pct']:.1f}%")
            m5.metric("Growth",  int(r["Growth"]),
                      delta=("▲" if r["Growth"] > 0 else "▼") if r["Growth"] != 0 else "—")


# ══════════════════════════════════════════════
# TAB 4 – ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📈 Analytics Dashboard</div>',
                unsafe_allow_html=True)

    if not st.session_state.sim_done:
        st.info("Run the simulation first.")
    else:
        df_stats = st.session_state.df_stats

        # Summary KPIs
        max_alive_gen = df_stats.loc[df_stats["Alive"].idxmax(), "Generation"]
        min_alive_gen = df_stats.loc[df_stats["Alive"].idxmin(), "Generation"]
        stability_pct = df_stats["Stable"].mean() * 100
        last = df_stats.iloc[-1]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Final Alive",      int(last["Alive"]))
        k2.metric("Final Dead",       int(last["Dead"]))
        k3.metric("Final Alive %",    f"{last['Alive_Pct']:.1f}%")
        k4.metric("Peak Alive Gen",   int(max_alive_gen))
        k5.metric("Min Alive Gen",    int(min_alive_gen))
        k6.metric("Stability %",      f"{stability_pct:.1f}%")

        st.markdown("---")

        # Charts 2x2
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_stat_chart(df_stats, "Alive", "Alive Cells over Generations", "#a0a0ff"))
            st.pyplot(plot_stat_chart(df_stats, "Growth", "Growth Rate (Δ Alive)", "#50d0a0"))
        with col_b:
            st.pyplot(plot_stat_chart(df_stats, "Dead", "Dead Cells over Generations", "#ff6060"))
            st.pyplot(plot_stat_chart(df_stats, "Alive_Pct", "Population % over Generations", "#ffa050"))

        st.markdown("**Full Statistics Table**")
        st.dataframe(df_stats.style.background_gradient(subset=["Alive", "Dead"], cmap="coolwarm"),
                     use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 – PATTERN DETECTION
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🧠 Advanced Pattern Detection</div>',
                unsafe_allow_html=True)

    if not st.session_state.sim_done:
        st.info("Run the simulation first.")
    else:
        df_stats = st.session_state.df_stats
        grids    = st.session_state.grids

        patterns = {}

        # Extinction
        extinct_rows = df_stats[df_stats["Alive"] == 0]
        patterns["Extinction"] = (
            f"⚠️ Extinction at Generation {int(extinct_rows.iloc[0]['Generation'])}"
            if not extinct_rows.empty else "✅ No extinction detected"
        )

        # Stable state
        stable_consecutive = 0
        stable_gen = None
        for i in range(1, len(grids)):
            if np.array_equal(grids[i], grids[i-1]):
                stable_consecutive += 1
                if stable_gen is None:
                    stable_gen = i
            else:
                stable_consecutive = 0
                stable_gen = None
        patterns["Stable State"] = (
            f"🔒 Stable from Generation {stable_gen}"
            if stable_gen else "🌀 No stable fixed-point detected"
        )

        # Oscillator (period-2)
        osc_gen = None
        for i in range(2, len(grids)):
            if np.array_equal(grids[i], grids[i-2]) and not np.array_equal(grids[i], grids[i-1]):
                osc_gen = i
                break
        patterns["Oscillator (Period-2)"] = (
            f"🔄 Oscillator detected near Generation {osc_gen}"
            if osc_gen else "➖ No period-2 oscillator detected"
        )

        # Population collapse (>50% drop in 5 gens)
        alive_series = df_stats["Alive"].values
        collapse_gen = None
        for i in range(5, len(alive_series)):
            if alive_series[i-5] > 0 and alive_series[i] / alive_series[i-5] < 0.5:
                collapse_gen = i + 1
                break
        patterns["Population Collapse"] = (
            f"📉 Collapse detected at Generation {collapse_gen}"
            if collapse_gen else "✅ No rapid population collapse"
        )

        # Max population
        max_gen = int(df_stats.loc[df_stats["Alive"].idxmax(), "Generation"])
        max_alive = int(df_stats["Alive"].max())
        patterns["Maximum Population"] = f"📈 Peak at Gen {max_gen} with {max_alive} alive cells"

        for name, finding in patterns.items():
            st.markdown(f"**{name}:** {finding}")

        st.markdown("---")
        st.markdown("**Population Trend Chart**")
        fig_pat, ax_pat = plt.subplots(figsize=(9, 3.5))
        fig_pat.patch.set_facecolor("#0f0c29")
        ax_pat.set_facecolor("#12122a")
        ax_pat.plot(df_stats["Generation"], df_stats["Alive"], "#a0a0ff", lw=2, label="Alive")
        ax_pat.plot(df_stats["Generation"], df_stats["Dead"],  "#ff6060", lw=2, label="Dead", alpha=0.6)
        ax_pat.axvline(max_gen, color="#ffd060", lw=1.5, linestyle="--", label=f"Peak Gen {max_gen}")
        if stable_gen:
            ax_pat.axvline(stable_gen, color="#50d080", lw=1.5, linestyle=":", label=f"Stable Gen {stable_gen}")
        ax_pat.legend(labelcolor="white", framealpha=0.3, fontsize=8)
        ax_pat.set_title("Full Population History", color="#c0c0ff")
        ax_pat.tick_params(colors="#8080aa")
        for sp in ax_pat.spines.values():
            sp.set_edgecolor("#333355")
        fig_pat.tight_layout()
        st.pyplot(fig_pat)


# ══════════════════════════════════════════════
# TAB 6 – DATA SCIENCE CONCEPTS
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">📚 Data Science Concepts Explained</div>',
                unsafe_allow_html=True)

    concepts = {
        "🎯 Thresholding": (
            "Thresholding converts continuous numeric values into binary categories. "
            "It is the foundation of many ML classifiers. Here, we decide whether a region "
            "is 'alive' (high-performing) or 'dead' (low-performing) based on a cutoff value."
        ),
        "🔢 Binary Classification": (
            "After thresholding, each data point becomes a 0 or 1. This mirrors supervised "
            "classification tasks like spam detection or disease diagnosis, where we output a "
            "binary decision from continuous input features."
        ),
        "🧬 Cellular Automata": (
            "A cellular automaton is a grid of cells, each evolving according to local rules. "
            "Conway's Game of Life is the most famous example. Despite simple rules, it can "
            "generate extraordinarily complex and unpredictable behaviour — a hallmark of "
            "complex systems."
        ),
        "✨ Emergent Behaviour": (
            "Emergence occurs when simple rules at the micro level produce complex patterns "
            "at the macro level. No single rule 'creates' a glider or oscillator — they arise "
            "from the collective interaction. This is analogous to how neurons form intelligence."
        ),
        "🔍 Pattern Recognition": (
            "Pattern recognition involves identifying recurring structures in data. In Conway's "
            "life, we detect stable blocks, oscillators, and gliders. In ML, CNNs detect edges, "
            "textures, and objects using learned spatial filters."
        ),
        "🗺 Spatial Data Analysis": (
            "Spatial analysis examines how geographic arrangement affects relationships. Grid-based "
            "cellular automata are natural spatial models, similar to raster GIS layers or image "
            "tensors — where the position of a cell affects its state and evolution."
        ),
    }

    for title, body in concepts.items():
        with st.expander(title, expanded=False):
            st.write(body)


# ══════════════════════════════════════════════
# TAB 7 – REAL WORLD INSIGHTS
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">🌍 Real-World Interpretations</div>',
                unsafe_allow_html=True)

    domains = [
        ("🏙 Urban Expansion", "Population_Density",
         "High-density zones (alive cells) surrounded by other dense zones keep growing — mirroring how cities attract more people to already crowded areas. Isolated dense cells shrink, like ghost towns."),
        ("🦠 Disease Spread", "Pollution_Index",
         "A high-pollution cell spreading to low-pollution neighbours models pathogen transmission. Reproduction (3-neighbour rule) = 'R₀ ≈ 3' herd transmission. Extinction = epidemic burnout."),
        ("🚗 Traffic Systems", "Employment_Rate",
         "High-employment cells represent busy intersections. Three busy neighbours creating a new busy cell models traffic congestion propagating outward from hubs at peak hours."),
        ("🌿 Ecological Systems", "Healthcare_Score",
         "Alive cells = healthy ecosystem zones. Underpopulation = habitat fragmentation. Overpopulation = carrying-capacity limits. Stable patterns = ecological equilibrium."),
        ("📱 Social Networks", "Internet_Penetration",
         "High-internet regions that cluster together grow further (viral adoption). Isolated high-internet cells die out (network effect). Three connected users create a new adopter — word-of-mouth marketing."),
        ("👥 Population Migration", "Income",
         "High-income zones attract neighbours (reproduction). Isolated high-income enclaves decline (underpopulation). Clusters of wealth persist and grow — wealth concentration dynamics."),
    ]

    for title, feat, interp in domains:
        with st.expander(f"{title} ← **{feat.replace('_',' ')}**"):
            st.write(interp)


# ══════════════════════════════════════════════
# TAB 8 – AI / ML CONNECTION
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">🤖 AI & Machine Learning Connections</div>',
                unsafe_allow_html=True)

    ai_topics = [
        ("🤖 Machine Learning", "Conway's life is a rule-based learner. The 'rules' are the model. ML replaces hand-crafted rules with learned ones. Both learn structure from local interactions — decision trees use local splits; CNNs use local receptive fields."),
        ("🧠 Deep Learning", "Convolutional Neural Networks (CNNs) apply learned 3×3 kernels across spatial grids — exactly how Conway counts 8 neighbours. Each convolution layer is a 'generation' of spatial transformation."),
        ("🕹 Agent-Based Modeling", "Each cell is an autonomous agent following local rules. ABM (used in epidemiology, economics, ecology) scales this to millions of heterogeneous agents with probabilistic rules."),
        ("🌀 Complex Systems", "Conway's life demonstrates sensitivity to initial conditions, self-organisation, and emergence — properties of all complex adaptive systems, from ant colonies to financial markets to neural networks."),
        ("🔄 Self-Organisation", "Stable patterns (blocks, oscillators) emerge without central control. Self-Organizing Maps (SOMs) in ML similarly cluster data by local competitive learning without supervision."),
        ("🧫 Artificial Life", "A-Life studies life-as-it-could-be. Conway's life is the foundational A-Life experiment. Modern Generative AI (LLMs, diffusion models) can be seen as artificial life: complex, self-sustaining, and emergent."),
    ]

    c1, c2 = st.columns(2)
    for i, (title, body) in enumerate(ai_topics):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"**{title}**")
        col.info(body)


# ══════════════════════════════════════════════
# TAB 9 – HINGLISH EXPLAINER
# ══════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">🇮🇳 Conway\'s Game of Life — Hinglish Mein Samjho</div>',
                unsafe_allow_html=True)

    hinglish_sections = [
        ("📊 Dataset kya represent karta hai?",
         "Humara dataset ek khayal ki duniya hai jisme 600 shaharon/regions ka data hai. "
         "Har row ek alag jagah ka data hai — wahan kitna pollution hai, income kitni hai, "
         "internet kitna chal raha hai. Yeh real-world data hai jo society ki ek tasveer dikhata hai."),

        ("🔄 Data ko cells mein kaise convert kiya?",
         "Humne ek simple rule follow kiya: ek threshold value set ki. "
         "Agar kisi region ka value us threshold se UPAR hai, toh woh cell 'Alive' (1) ho gaya. "
         "Agar NEECHE hai, toh 'Dead' (0). Bas ek line decide karti hai kaun jeega kaun marega! "
         "Yeh binary thresholding hai — ML ka sabse basic concept."),

        ("🟣⚫ Alive aur Dead cells kya hote hain?",
         "Alive cell = ek aacha/thriving region. Jaise zyada income, achhi healthcare, ya high internet penetration. "
         "Dead cell = ek struggling region. Sochon jaise ek gaon jahan employment nahi, pollution zyada hai. "
         "Grid pe, blue/purple = alive (thriving), black = dead (struggling)."),

        ("📏 Conway Rules ka intuition (Aasaan Bhasha Mein)",
         "Rule 1 – Akela rahega toh marta hai: Ek alive cell ke paas sirf 1 neighbour hai — woh cell survive nahi kar sakta. "
         "Jaise ek akela doctor poore district ke liye kaam nahi kar sakta.\n\n"
         "Rule 2 – Balance mein zindagi: 2-3 neighbours ke saath cell jeeta rehta hai. "
         "Yeh healthy community hai — naa zyada, naa kam.\n\n"
         "Rule 3 – Bheed mein daba ke mar gaya: Agar 4 ya zyada neighbours hain toh resources khatam — cell dies. "
         "Overpopulation = resource depletion.\n\n"
         "Rule 4 – Paida hona: Ek dead cell ke exactly 3 alive neighbours hain toh woh born ho jata hai. "
         "Yeh reproduction hai — critical mass reach hone par growth hoti hai."),

        ("🌟 Patterns kyu bante hain?",
         "Yeh magic hai! Simple 4 rules se complex patterns emerge hote hain — stable blocks, oscillators, gliders. "
         "Koi plan nahi tha, koi central control nahi tha — phir bhi pattern bana. "
         "Yeh 'emergent behaviour' hai. Jaise traffic jam plan nahi hota, phir bhi consistent pattern follow karta hai."),

        ("🌍 Real World Mein Iska Use",
         "Disease spreading: Corona ka phelna bhi aisa hi tha — infected cells (R₀=3) new cells create karte hain. "
         "Urban growth: Shehar expand hote hain jab neighbouring areas develop hote hain. "
         "Ecosystem: Van ki cutting isolated patches karti hai — underpopulation se extinction. "
         "Social media viral: Ek post teen logon tak pahunchi → naya 'alive' user born hua!"),

        ("🤖 Data Science aur AI Mein Connection",
         "Thresholding → Binary Classification (SVM, Logistic Regression ka foundation)\n"
         "Neighbour counting → Convolution (CNN ka dil)\n"
         "Emergent patterns → Deep Learning ka unexplained generalisation\n"
         "Stable states → Convergence in optimization\n"
         "Self-organisation → Unsupervised Clustering (k-means, SOMs)\n"
         "Conway ka grid → Ek 2D tensor jo har generation pe transform hota hai (exactly like CNN forward pass!)"),
    ]

    for title, body in hinglish_sections:
        st.markdown(f'<div class="hinglish-card"><strong>{title}</strong><br><br>{body.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)
        st.markdown("")


# ══════════════════════════════════════════════
# TAB 10 – REPORT
# ══════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-header">📄 Downloadable Report</div>',
                unsafe_allow_html=True)

    if not st.session_state.sim_done:
        st.info("Run the simulation to generate a report.")
    else:
        df_stats  = st.session_state.df_stats
        ig        = st.session_state.initial_grid
        thr_used  = st.session_state.threshold_used
        last_stat = df_stats.iloc[-1]
        max_row   = df_stats.loc[df_stats["Alive"].idxmax()]
        stable_pct = df_stats["Stable"].mean() * 100

        lines = [
            "=" * 60,
            "  CONWAY'S GAME OF LIFE — DATA SCIENCE SIMULATION REPORT",
            "=" * 60,
            "",
            "── DATASET SUMMARY ──────────────────────────────────────",
            f"  Records            : {len(df)}",
            f"  Features           : {list(df.columns)}",
            f"  Missing Values     : {int(df.isnull().sum().sum())}",
            "",
            "── CONVERSION SETTINGS ──────────────────────────────────",
            f"  Feature Selected   : {feature}",
            f"  Threshold Strategy : {threshold_mode}",
            f"  Threshold Value    : {thr_used:.4f}",
            f"  Grid Size          : {grid_size} × {grid_size}",
            "",
            "── SIMULATION SETTINGS ──────────────────────────────────",
            f"  Generations Run    : {len(df_stats)}",
            f"  Random Seed        : {seed}",
            "",
            "── INITIAL STATE ────────────────────────────────────────",
            f"  Initial Alive Cells: {int(ig.sum())}",
            f"  Initial Dead Cells : {int(ig.size - ig.sum())}",
            f"  Initial Alive %    : {ig.mean()*100:.2f}%",
            "",
            "── FINAL STATE ──────────────────────────────────────────",
            f"  Final Alive Cells  : {int(last_stat['Alive'])}",
            f"  Final Dead Cells   : {int(last_stat['Dead'])}",
            f"  Final Alive %      : {last_stat['Alive_Pct']:.2f}%",
            "",
            "── PATTERN DISCOVERIES ──────────────────────────────────",
            f"  Peak Population    : Gen {int(max_row['Generation'])} with {int(max_row['Alive'])} alive cells",
            f"  Stability Score    : {stable_pct:.1f}% of generations were stable",
            f"  Extinction         : {'YES' if (df_stats['Alive']==0).any() else 'NO'}",
            "",
            "── KEY OBSERVATIONS ─────────────────────────────────────",
            "  1. Thresholding converts continuous feature data into",
            "     binary alive/dead states — the basis of binary ML.",
            "  2. Conway rules simulate emergent societal dynamics.",
            "  3. Stable patterns indicate equilibrium in the dataset.",
            "  4. Extinction events signal critical data thresholds.",
            "  5. Neighbour counting mirrors CNN convolution operations.",
            "",
            "=" * 60,
            "  Generated by: Conway's Game of Life — Data Science App",
            "=" * 60,
        ]

        report_text = "\n".join(lines)
        st.text(report_text)

        st.download_button(
            label="⬇️  Download Report (.txt)",
            data=report_text,
            file_name="conway_simulation_report.txt",
            mime="text/plain",
        )

        # CSV export
        csv = df_stats.to_csv(index=False)
        st.download_button(
            label="⬇️  Download Statistics CSV",
            data=csv,
            file_name="conway_stats.csv",
            mime="text/csv",
        )