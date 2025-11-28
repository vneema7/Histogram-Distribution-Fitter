import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------
# PAGE / STYLE
# ---------------------------
st.set_page_config(page_title="Histogram Fitter", layout="wide")
st.title("Histogram + Distribution Fitter")

st.markdown("""
<style>
section[data-testid="stSidebar"] * {
    font-size: 16px;
    font-family: 'Roboto', sans-serif;
}
.small-note { font-size: 13px; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1) DATA INPUT (SIDEBAR)
# ---------------------------
with st.sidebar:
    st.header("Data Input")

    mode = st.radio("Choose input method:", ["Manual entry", "Upload CSV"])
    data = None

    if mode == "Manual entry":
        raw = st.text_area(
            "Enter numbers separated by commas/spaces/newlines:",
            "1,2,3,4,5,6,7,8,9"
        )
        try:
            tokens = raw.replace(",", " ").split()
            data = np.array([float(x) for x in tokens], dtype=float)
        except Exception:
            st.error("Couldn't parse numbers. Fix your input.")
            data = None
    else:
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            st.write("Preview:", df.head())
            col = st.selectbox("Pick a column to fit:", df.columns)
            data = df[col].dropna().to_numpy(dtype=float)

if data is None or len(data) == 0:
    st.stop()

# ---------------------------
# 2) DISTRIBUTIONS (>=10)
# ---------------------------
DIST_MAP = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Weibull (min)": stats.weibull_min,
    "Weibull (max)": stats.weibull_max,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Student-t": stats.t,
    "Chi-square": stats.chi2,
    "Pareto": stats.pareto,
    "Rayleigh": stats.rayleigh,
}

# ---------------------------
# SESSION STATE DEFAULTS
# ---------------------------
if "bg_color_name" not in st.session_state:
    st.session_state.bg_color_name = "White"
if "hist_color_name" not in st.session_state:
    st.session_state.hist_color_name = "Skyblue"
if "line_color_name" not in st.session_state:
    st.session_state.line_color_name = "Red"
if "auto_params" not in st.session_state:
    st.session_state.auto_params = None
if "dist_name" not in st.session_state:
    st.session_state.dist_name = list(DIST_MAP.keys())[0]
if "bins" not in st.session_state:
    st.session_state.bins = 30
if "last_dist_name" not in st.session_state:
    st.session_state.last_dist_name = st.session_state.dist_name

COLOR_MAP = {
    "White": "white",
    "Red": "red",
    "Blue": "blue",
    "Green": "green",
    "Orange": "orange",
    "Purple": "purple",
    "Pink": "pink",
    "Cyan": "cyan",
    "Black": "black",
    "Skyblue": "skyblue",
    "Lightgreen": "lightgreen",
    "Yellow": "yellow",
    "Gray": "gray",
}

# ---------------------------
# GLOBAL CONTROLS (MAIN)
# ---------------------------
top_left, top_right = st.columns([1.2, 1])
with top_left:
    st.subheader("Distribution Selection")
    dist_choice = st.selectbox(
        "Choose distribution to fit:",
        list(DIST_MAP.keys()),
        index=list(DIST_MAP.keys()).index(st.session_state.dist_name)
    )

    # If user changed the distribution, reset auto_params
    if dist_choice != st.session_state.last_dist_name:
        st.session_state.auto_params = None  # kills old fit
        st.session_state.last_dist_name = dist_choice

    st.session_state.dist_name = dist_choice

with top_right:
    st.subheader("Histogram Control")
    st.session_state.bins = st.slider("Histogram bins", 5, 150, st.session_state.bins)

dist_name = st.session_state.dist_name
dist = DIST_MAP[dist_name]
bins = st.session_state.bins

# ---------------------------
# TABS
# ---------------------------
plot_tab, cus_tab, manual_tab = st.tabs(["PLOT", "CUSTOMIZE", "MANUAL FITTING"])

# ---------------------------
# CUSTOMIZE TAB
# ---------------------------
with cus_tab:
    st.subheader("Customization")

    bg_options = ["White", "Red", "Blue", "Green", "Orange", "Purple", "Pink", "Cyan", "Black"]
    hist_options = ["Skyblue", "Lightgreen", "Orange", "Yellow", "Gray", "Purple", "Red", "Blue", "Black"]
    line_options = ["Red", "Blue", "Green", "Orange", "Purple", "Black"]

    st.session_state.bg_color_name = st.selectbox(
        "Plot Background Color",
        bg_options,
        index=bg_options.index(st.session_state.bg_color_name)
    )

    st.session_state.hist_color_name = st.selectbox(
        "Histogram Bar Color",
        hist_options,
        index=hist_options.index(st.session_state.hist_color_name)
    )

    st.session_state.line_color_name = st.selectbox(
        "Fit Line Color",
        line_options,
        index=line_options.index(st.session_state.line_color_name)
    )

# Resolve to real matplotlib colors
bg_color = COLOR_MAP[st.session_state.bg_color_name]
hist_color = COLOR_MAP[st.session_state.hist_color_name]
line_color = COLOR_MAP[st.session_state.line_color_name]

# ---------------------------
# PLOT TAB (AUTO FIT)
# ---------------------------
with plot_tab:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Auto-Fit Settings")

        if st.button("Auto-fit distribution"):
            try:
                st.session_state.auto_params = dist.fit(data)
            except Exception as e:
                st.session_state.auto_params = None
                st.error(f"Fit failed: {e}")

        st.markdown(
            "<div class='small-note'>Tip: Auto-fit finds the best SciPy parameters. "
            "Use Manual Fitting tab to tweak.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Auto-Fit Plot")

        auto_params = st.session_state.auto_params

        # NEW: no graph unless Auto-fit has run successfully
        if auto_params is None:
            st.info("Select a distribution and click **Auto-fit distribution** to see the plot.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor("white")

            ax.hist(
                data, bins=bins, density=True,
                alpha=0.65, edgecolor="black",
                color=hist_color
            )

            x = np.linspace(data.min(), data.max(), 600)

            try:
                pdf = dist.pdf(x, *auto_params)
                ax.plot(x, pdf, color=line_color, linewidth=2.5, label=f"{dist_name} auto-fit")
                ax.legend()
            except Exception as e:
                st.error(f"Error plotting auto-fit: {e}")

            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            st.pyplot(fig)

# ---------------------------
# 3) FIT RESULTS + QUALITY
# ---------------------------
st.subheader("Fit Results")

auto_params = st.session_state.auto_params
if auto_params is None:
    st.info("Run **Auto-fit distribution** to see parameters and fit quality.")
else:
    st.write("Auto-fit parameters (shape(s), loc, scale):")
    st.code(auto_params)

    # Histogram density vs PDF at bin centers
    hist_y, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    try:
        pdf_centers = dist.pdf(centers, *auto_params)
    except Exception as e:
        st.error(f"Error evaluating PDF at bin centers: {e}")
        pdf_centers = np.zeros_like(centers)

    err = np.abs(hist_y - pdf_centers)
    mae = float(err.mean())
    max_err = float(err.max())

    # KS test (distributional goodness)
    try:
        ks_stat, ks_p = stats.kstest(data, dist.cdf, args=auto_params)
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    # Log-likelihood + AIC/BIC
    try:
        ll = float(np.sum(dist.logpdf(data, *auto_params)))
        k = len(auto_params)
        n = len(data)
        aic = float(2 * k - 2 * ll)
        bic = float(k * np.log(n) - 2 * ll)
    except Exception:
        ll, aic, bic = np.nan, np.nan, np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean abs error", f"{mae:.5f}")
    m2.metric("Max abs error", f"{max_err:.5f}")
    m3.metric("KS statistic", "—" if np.isnan(ks_stat) else f"{ks_stat:.5f}")
    m4.metric("KS p-value", "—" if np.isnan(ks_p) else f"{ks_p:.5f}")

    q1, q2, q3 = st.columns(3)
    q1.metric("Log-likelihood", "—" if np.isnan(ll) else f"{ll:.3f}")
    q2.metric("AIC (lower better)", "—" if np.isnan(aic) else f"{aic:.3f}")
    q3.metric("BIC (lower better)", "—" if np.isnan(bic) else f"{bic:.3f}")

    with st.sidebar:
        st.header("Fit Quality (quick view)")
        st.write(f"Mean abs error: **{mae:.5f}**")
        st.write(f"Max error: **{max_err:.5f}**")
        if not np.isnan(ks_stat):
            st.write(f"KS stat: **{ks_stat:.5f}**")
            st.write(f"KS p-value: **{ks_p:.5f}**")
        if not np.isnan(aic):
            st.write(f"AIC: **{aic:.2f}**")
            st.write(f"BIC: **{bic:.2f}**")

# ---------------------------
# 4) MANUAL FITTING
# ---------------------------
with manual_tab:
    st.subheader("Manual Fitting (sliders)")
    st.markdown("<div class='small-note'>Sliders use the current auto-fit as a starting point.</div>", unsafe_allow_html=True)

    auto_params = st.session_state.auto_params

    if auto_params is None:
        st.info("Run **Auto-fit distribution** first, then come back here for manual tuning.")
    else:
        base_params = auto_params

        shapes = base_params[:-2]
        loc0, scale0 = base_params[-2], base_params[-1]

        manual_shapes = []
        shape_names = [f"shape_{i+1}" for i in range(len(shapes))]

        cA, cB = st.columns(2)

        with cA:
            for name, val in zip(shape_names, shapes):
                v = float(val)
                span = max(1.0, 5.0 * abs(v))
                low = v - span
                high = v + span
                manual_shapes.append(
                    st.slider(name, low, high, v)
                )

        with cB:
            std = float(np.std(data)) if np.std(data) > 0 else 1.0
            manual_loc = st.slider("loc", float(loc0 - 5 * std), float(loc0 + 5 * std), float(loc0))
            manual_scale = st.slider(
                "scale",
                1e-6,
                float(max(scale0 * 5, 1e-3)),
                float(max(scale0, 1e-3))
            )

        manual_params = tuple(manual_shapes + [manual_loc, manual_scale])

        # Manual plot with same styling
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.set_facecolor(bg_color)
        fig2.patch.set_facecolor("white")

        ax2.hist(
            data, bins=bins, density=True,
            alpha=0.65, edgecolor="black",
            color=hist_color
        )

        x = np.linspace(data.min(), data.max(), 600)
        try:
            ax2.plot(
                x, dist.pdf(x, *manual_params),
                color=line_color, linewidth=2.5,
                label=f"{dist_name} manual"
            )
        except Exception as e:
            st.error(f"Error plotting manual fit: {e}")

        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        st.pyplot(fig2)

        st.write("Manual parameters (shape(s), loc, scale):")
        st.code(manual_params)
