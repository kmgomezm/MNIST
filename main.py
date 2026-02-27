import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    ShuffleSplit, LeaveOneOut, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from PIL import Image
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    :root {
        --bg: #0d0d0f;
        --surface: #16161a;
        --surface2: #1e1e24;
        --border: #2a2a35;
        --accent: #7c6af7;
        --accent2: #f7a26a;
        --text: #e8e8f0;
        --muted: #888899;
        --success: #5af7a2;
        --danger: #f76a6a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
        color: var(--text);
    }

    .metric-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent); }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent);
        border-left: 3px solid var(--accent);
        padding-left: 10px;
        margin-bottom: 16px;
    }

    .stButton>button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 10px 24px;
        cursor: pointer;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }

    .stSelectbox>div, .stSlider>div { color: var(--text); }

    .result-box {
        background: linear-gradient(135deg, var(--surface2), var(--surface));
        border: 2px solid var(--accent);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .result-digit {
        font-family: 'Space Mono', monospace;
        font-size: 4rem;
        color: var(--accent);
        line-height: 1;
    }
    .result-label {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .tag {
        display: inline-block;
        background: var(--accent);
        color: white;
        border-radius: 99px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }

    div[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    .stDataFrame { background: var(--surface2); }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: var(--muted); font-family: 'Space Mono', monospace; font-size: 0.78rem; }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
CLASSIFIERS = {
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}

CV_STRATEGIES = {
    "Stratified K-Fold (k=5)": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    "Stratified K-Fold (k=10)": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    "ShuffleSplit (10 iters)": ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
}

PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = ["#7c6af7", "#f7a26a", "#5af7a2", "#f76a6a", "#6af0f7", "#f76af0"]


@st.cache_data
def load_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    return X, y, digits


def build_pipeline(clf, use_pca, n_components):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = None
    return {"Accuracy": acc, "F1 (weighted)": f1, "Precision": prec, "Recall": rec, "ROC AUC": auc}


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        labels=dict(x="Predicted", y="Actual"),
        title=title, template=PLOTLY_TEMPLATE,
        aspect="auto"
    )
    fig.update_layout(font_family="Space Mono", title_font_size=14, height=400)
    return fig


def plot_pca_variance(X_scaled, max_components=50):
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, max_components + 1)),
        y=pca.explained_variance_ratio_ * 100,
        name="Individual", marker_color="#7c6af7", opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, max_components + 1)),
        y=cumvar, name="Cumulative",
        line=dict(color="#f7a26a", width=2), mode="lines+markers", marker_size=4
    ))
    fig.add_hline(y=95, line_dash="dash", line_color="#5af7a2",
                  annotation_text="95% variance")
    fig.update_layout(
        title="PCA – Explained Variance", template=PLOTLY_TEMPLATE,
        xaxis_title="Component", yaxis_title="Explained Variance (%)",
        font_family="Space Mono", height=380, legend=dict(orientation="h")
    )
    return fig


def metric_cards(metrics, split="Test"):
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        val = f"{v:.4f}" if v is not None else "N/A"
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{k}<br><small>{split}</small></div>
        </div>""", unsafe_allow_html=True)


# ─── App ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Space Mono;font-size:2rem;margin-bottom:0;'>
🔢 MNIST Digit Classifier
</h1>
<p style='color:#888899;font-size:0.9rem;margin-top:6px;'>
Multiclass classification · 10 classes · Scikit-learn digits dataset
</p>
""", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown('<p class="section-header">Dataset Split</p>', unsafe_allow_html=True)
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100

    st.markdown('<p class="section-header">PCA</p>', unsafe_allow_html=True)
    use_pca = st.toggle("Enable PCA", value=False)
    n_components = st.slider("PCA components", 5, 60, 30, 5, disabled=not use_pca)

    st.markdown('<p class="section-header">Models</p>', unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Select classifiers",
        list(CLASSIFIERS.keys()),
        default=["Naive Bayes", "K-Nearest Neighbors", "Support Vector Machine",
                 "Random Forest", "Decision Tree"]
    )

    st.markdown('<p class="section-header">Cross-Validation</p>', unsafe_allow_html=True)
    cv_strategy_name = st.selectbox("CV strategy", list(CV_STRATEGIES.keys()))

    run_btn = st.button("🚀 Train & Evaluate")

# ─── Load Data ────────────────────────────────────────────────────────────────
X, y, digits = load_data()

tabs = st.tabs(["📊 Data Quality", "🧪 Model Results", "📈 Cross-Validation", "✏️ Draw & Predict"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Quality
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1, c2, c3, c4], [
        ("Total Samples", len(X)),
        ("Features", X.shape[1]),
        ("Classes", len(np.unique(y))),
        ("Missing Values", int(np.isnan(X).sum()))
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Class Distribution</p>', unsafe_allow_html=True)
        counts = np.bincount(y)
        fig_dist = px.bar(
            x=list(range(10)), y=counts,
            labels={"x": "Digit", "y": "Count"},
            color=counts, color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE, title="Samples per Class"
        )
        fig_dist.update_layout(font_family="Space Mono", showlegend=False, height=320,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Feature Statistics</p>', unsafe_allow_html=True)
        df_stats = pd.DataFrame({
            "Min": X.min(axis=0), "Max": X.max(axis=0),
            "Mean": X.mean(axis=0), "Std": X.std(axis=0)
        })
        fig_box = px.box(
            pd.DataFrame(X[:, :16], columns=[f"px_{i}" for i in range(16)]),
            template=PLOTLY_TEMPLATE, title="Feature Value Distribution (first 16 features)"
        )
        fig_box.update_layout(font_family="Space Mono", height=320, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="section-header">Sample Digits</p>', unsafe_allow_html=True)
    cols_imgs = st.columns(10)
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        img = digits.images[idx]
        fig_img, ax = plt.subplots(figsize=(1.2, 1.2), facecolor="#16161a")
        ax.imshow(img, cmap="magma", interpolation="nearest")
        ax.axis("off")
        ax.set_title(str(digit), color="#7c6af7", fontsize=10, pad=2, fontfamily="monospace")
        cols_imgs[digit].pyplot(fig_img, use_container_width=True)
        plt.close()

    st.markdown('<p class="section-header">PCA Variance Analysis</p>', unsafe_allow_html=True)
    scaler_pca = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(X)
    st.plotly_chart(plot_pca_variance(X_scaled_pca), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Session state for trained models
# ════════════════════════════════════════════════════════════════════════════
if "trained" not in st.session_state:
    st.session_state.trained = {}

if run_btn and selected_models:
    with st.spinner("Training models…"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        results = {}
        for name in selected_models:
            clf = CLASSIFIERS[name]
            pipe = build_pipeline(clf, use_pca, n_components)
            pipe.fit(X_train, y_train)

            y_pred_train = pipe.predict(X_train)
            y_pred_test = pipe.predict(X_test)

            try:
                y_prob_test = pipe.predict_proba(X_test)
            except Exception:
                y_prob_test = None

            train_metrics = compute_metrics(y_train, y_pred_train)
            test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)

            # Cross-validation
            cv = CV_STRATEGIES[cv_strategy_name]
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

            results[name] = {
                "pipe": pipe,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_scores": cv_scores,
                "y_pred_test": y_pred_test,
                "y_test": y_test,
                "y_pred_train": y_pred_train,
                "y_train": y_train,
            }

        st.session_state.trained = results
        st.session_state.config = {
            "use_pca": use_pca, "n_components": n_components,
            "test_size": test_size, "cv_strategy": cv_strategy_name
        }
    st.success("✅ Training complete!")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Results
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if not st.session_state.trained:
        st.info("👈 Configure and click **Train & Evaluate** in the sidebar.")
    else:
        results = st.session_state.trained
        cfg = st.session_state.config

        # Summary table
        st.markdown('<p class="section-header">Summary – All Models</p>', unsafe_allow_html=True)
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Train Acc": f"{r['train_metrics']['Accuracy']:.4f}",
                "Test Acc": f"{r['test_metrics']['Accuracy']:.4f}",
                "Test F1": f"{r['test_metrics']['F1 (weighted)']:.4f}",
                "Test Prec": f"{r['test_metrics']['Precision']:.4f}",
                "Test Recall": f"{r['test_metrics']['Recall']:.4f}",
                "ROC AUC": f"{r['test_metrics']['ROC AUC']:.4f}" if r['test_metrics']['ROC AUC'] else "N/A",
                "CV Mean Acc": f"{r['cv_scores'].mean():.4f}",
                "CV Std": f"{r['cv_scores'].std():.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        model_names = list(results.keys())
        test_accs = [r["test_metrics"]["Accuracy"] for r in results.values()]
        train_accs = [r["train_metrics"]["Accuracy"] for r in results.values()]
        cv_means = [r["cv_scores"].mean() for r in results.values()]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Train Acc", x=model_names, y=train_accs,
                                   marker_color="#7c6af7", opacity=0.8))
        fig_comp.add_trace(go.Bar(name="Test Acc", x=model_names, y=test_accs,
                                   marker_color="#f7a26a"))
        fig_comp.add_trace(go.Scatter(name="CV Mean Acc", x=model_names, y=cv_means,
                                       mode="markers+lines", marker=dict(color="#5af7a2", size=10),
                                       line=dict(color="#5af7a2", dash="dot")))
        fig_comp.update_layout(
            barmode="group", template=PLOTLY_TEMPLATE,
            font_family="Space Mono", height=380,
            yaxis=dict(range=[0, 1.05], title="Score"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Per-model detail
        st.markdown('<p class="section-header">Detailed Results per Model</p>', unsafe_allow_html=True)
        selected_detail = st.selectbox("Select model to inspect", list(results.keys()))
        r = results[selected_detail]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Metrics**")
            metric_cards(r["train_metrics"], "Train")
        with col2:
            st.markdown("**Test Metrics**")
            metric_cards(r["test_metrics"], "Test")

        st.markdown("<br>", unsafe_allow_html=True)
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.plotly_chart(
                plot_confusion_matrix(r["y_train"], r["y_pred_train"], f"{selected_detail} – Train CM"),
                use_container_width=True
            )
        with cm_col2:
            st.plotly_chart(
                plot_confusion_matrix(r["y_test"], r["y_pred_test"], f"{selected_detail} – Test CM"),
                use_container_width=True
            )

        # Classification report
        st.markdown('<p class="section-header">Classification Report (Test)</p>', unsafe_allow_html=True)
        report = classification_report(r["y_test"], r["y_pred_test"], output_dict=True)
        report_df = pd.DataFrame(report).T.iloc[:-3]
        report_df = report_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        st.dataframe(report_df, use_container_width=True)

        # PCA/No-PCA toggle info
        st.markdown(f"""
        <div style='background:#1e1e24;border:1px solid #2a2a35;border-radius:10px;padding:12px 18px;font-size:0.85rem;color:#888899;'>
        ⚙️ Config: <b style='color:#7c6af7;'>PCA={'ON – ' + str(cfg['n_components']) + ' components' if cfg['use_pca'] else 'OFF'}</b> 
        &nbsp;|&nbsp; Test size: <b style='color:#f7a26a;'>{int(cfg['test_size']*100)}%</b>
        &nbsp;|&nbsp; CV: <b style='color:#5af7a2;'>{cfg['cv_strategy']}</b>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Cross-Validation
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if not st.session_state.trained:
        st.info("👈 Configure and click **Train & Evaluate** in the sidebar.")
    else:
        results = st.session_state.trained
        st.markdown('<p class="section-header">Cross-Validation Scores per Model</p>', unsafe_allow_html=True)

        # Box plots
        all_scores = {name: r["cv_scores"] for name, r in results.items()}
        fig_cv = go.Figure()
        for i, (name, scores) in enumerate(all_scores.items()):
            fig_cv.add_trace(go.Box(
                y=scores, name=name, marker_color=PALETTE[i % len(PALETTE)],
                boxmean="sd"
            ))
        fig_cv.update_layout(
            template=PLOTLY_TEMPLATE, font_family="Space Mono",
            yaxis_title="CV Accuracy", height=420,
            title="Cross-Validation Score Distribution"
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        # Table
        st.markdown('<p class="section-header">CV Statistics</p>', unsafe_allow_html=True)
        cv_rows = []
        for name, r in results.items():
            s = r["cv_scores"]
            cv_rows.append({
                "Model": name,
                "Mean": f"{s.mean():.4f}",
                "Std": f"{s.std():.4f}",
                "Min": f"{s.min():.4f}",
                "Max": f"{s.max():.4f}",
                "95% CI Low": f"{s.mean() - 1.96*s.std():.4f}",
                "95% CI High": f"{s.mean() + 1.96*s.std():.4f}",
            })
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        # Fold-by-fold detail
        st.markdown('<p class="section-header">Fold-by-Fold Scores</p>', unsafe_allow_html=True)
        model_cv = st.selectbox("Select model for fold detail", list(results.keys()), key="cv_model")
        scores = results[model_cv]["cv_scores"]
        fig_folds = go.Figure()
        fig_folds.add_trace(go.Scatter(
            x=list(range(1, len(scores) + 1)), y=scores,
            mode="lines+markers+text", text=[f"{s:.3f}" for s in scores],
            textposition="top center", marker=dict(color="#7c6af7", size=10),
            line=dict(color="#7c6af7"), name="Fold Acc"
        ))
        fig_folds.add_hline(y=scores.mean(), line_dash="dash",
                             line_color="#f7a26a", annotation_text=f"Mean={scores.mean():.4f}")
        fig_folds.update_layout(
            template=PLOTLY_TEMPLATE, font_family="Space Mono",
            xaxis_title="Fold", yaxis_title="Accuracy",
            title=f"{model_cv} – Per-Fold CV Scores", height=350
        )
        st.plotly_chart(fig_folds, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Draw & Predict
# ════════════════════════════════════════════════════════════════════════════

def preprocess_canvas_to_mnist(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a canvas RGBA image (H×W×4) into an 8×8 float array matching
    the sklearn digits dataset distribution (values 0–16).

    Pipeline:
      1. RGBA → Grayscale
      2. Crop to bounding box of the drawn stroke (removes empty margins)
      3. Add padding (20%) so the digit doesn't touch the edges — matches MNIST style
      4. Resize to 8×8 with high-quality downsampling
      5. Invert if needed so digit is bright on dark background
      6. Rescale linearly to [0, 16] range
    Returns (img_flat_1d, img_8x8) tuple.
    """
    from PIL import ImageOps, ImageFilter

    img_pil = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    img_np = np.array(img_pil)

    # ── 1. Check canvas has actual content ──────────────────────────────────
    if img_np.max() == 0:
        return None, None

    # ── 2. Crop bounding box around the drawn stroke ─────────────────────────
    # Find rows/cols that have any non-zero pixel
    rows = np.any(img_np > 10, axis=1)
    cols = np.any(img_np > 10, axis=0)
    if not rows.any() or not cols.any():
        return None, None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_np[rmin:rmax+1, cmin:cmax+1]

    # ── 3. Pad to square with 20% margin (mirrors MNIST centering) ───────────
    h, w = cropped.shape
    side = max(h, w)
    pad = max(int(side * 0.25), 2)
    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    y_off = pad + (side - h) // 2
    x_off = pad + (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cropped

    # ── 4. Smooth slightly before aggressive downscale ───────────────────────
    img_sq = Image.fromarray(square)
    img_sq = img_sq.filter(ImageFilter.GaussianBlur(radius=1))

    # ── 5. Resize to 8×8 ─────────────────────────────────────────────────────
    img_8 = img_sq.resize((8, 8), Image.LANCZOS)
    img_8_np = np.array(img_8, dtype=float)

    # ── 6. Rescale to [0, 16] matching sklearn digits range ─────────────────
    mn, mx = img_8_np.min(), img_8_np.max()
    if mx > mn:
        img_8_np = (img_8_np - mn) / (mx - mn) * 16.0
    else:
        img_8_np = np.zeros_like(img_8_np)

    return img_8_np.flatten().reshape(1, -1), img_8_np


with tabs[3]:
    st.markdown('<p class="section-header">Draw a Digit & Predict</p>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.info("👈 Train at least one model first, then come back here.")
    else:
        try:
            from streamlit_drawable_canvas import st_canvas

            st.markdown("""
            <div style='background:#1e1e24;border:1px solid #2a2a35;border-radius:10px;
            padding:10px 16px;font-size:0.82rem;color:#888899;margin-bottom:16px;'>
            💡 <b style='color:#f7a26a;'>Tips para mejor precisión:</b>
            Dibuja el dígito <b>grande y centrado</b>, con trazos gruesos.
            El sistema recortará automáticamente los márgenes y ajustará el tamaño a 8×8 px.
            </div>""", unsafe_allow_html=True)

            col_draw, col_result = st.columns([1.1, 1])

            with col_draw:
                st.markdown("**Dibuja un dígito (0–9):**")
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=22,          # trazo más grueso = más píxeles al reducir
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                )

                predict_model = st.selectbox(
                    "Modelo para predicción",
                    list(st.session_state.trained.keys())
                )
                predict_btn = st.button("🔍 Predecir dígito")

            with col_result:
                if predict_btn and canvas_result.image_data is not None:
                    img_flat, img_8x8 = preprocess_canvas_to_mnist(canvas_result.image_data)

                    if img_flat is None:
                        st.warning("⚠️ Canvas vacío — dibuja un dígito primero.")
                    else:
                        pipe = st.session_state.trained[predict_model]["pipe"]
                        prediction = pipe.predict(img_flat)[0]

                        try:
                            proba = pipe.predict_proba(img_flat)[0]
                            conf = proba.max()
                        except Exception:
                            proba = None
                            conf = None

                        st.markdown(f"""
                        <div class="result-box">
                            <div class="result-digit">{prediction}</div>
                            <div class="result-label">Dígito predicho</div>
                            {'<div style="color:#5af7a2;font-family:Space Mono;margin-top:12px;font-size:1.1rem;">Confianza: ' + f'{conf:.1%}</div>' if conf else ''}
                        </div>""", unsafe_allow_html=True)

                        # ── Visualización del pipeline de preprocesamiento ──
                        st.markdown("<br>**Pipeline de preprocesamiento:**", unsafe_allow_html=True)

                        # Mostrar: original → 8x8 → valores numéricos
                        fig_pipe, axes = plt.subplots(1, 3, figsize=(7, 2.5),
                                                       facecolor="#0d0d0f")
                        fig_pipe.patch.set_facecolor("#0d0d0f")

                        # Original (canvas completo)
                        orig_gray = Image.fromarray(
                            canvas_result.image_data.astype("uint8"), mode="RGBA"
                        ).convert("L")
                        axes[0].imshow(np.array(orig_gray), cmap="gray", interpolation="nearest")
                        axes[0].set_title("Canvas original", color="#888899",
                                           fontsize=8, fontfamily="monospace")
                        axes[0].axis("off")

                        # 8×8 procesada
                        axes[1].imshow(img_8x8, cmap="magma", interpolation="nearest",
                                        vmin=0, vmax=16)
                        axes[1].set_title("8×8 (input modelo)", color="#7c6af7",
                                           fontsize=8, fontfamily="monospace")
                        axes[1].axis("off")

                        # Heatmap con valores numéricos (0–16)
                        im = axes[2].imshow(img_8x8, cmap="magma", interpolation="nearest",
                                             vmin=0, vmax=16)
                        for row in range(8):
                            for col in range(8):
                                val = img_8x8[row, col]
                                axes[2].text(col, row, f"{val:.0f}",
                                              ha="center", va="center",
                                              fontsize=5, color="white" if val > 6 else "#555")
                        axes[2].set_title("Valores [0–16]", color="#f7a26a",
                                           fontsize=8, fontfamily="monospace")
                        axes[2].axis("off")

                        plt.tight_layout(pad=0.5)
                        st.pyplot(fig_pipe)
                        plt.close()

                        # Comparar con dígitos reales del dataset
                        st.markdown("**Comparar con ejemplos del dataset:**",
                                    unsafe_allow_html=True)
                        sample_cols = st.columns(5)
                        sample_idxs = np.where(y == prediction)[0][:5]
                        for sc, si in zip(sample_cols, sample_idxs):
                            fig_s, ax_s = plt.subplots(figsize=(1, 1), facecolor="#16161a")
                            ax_s.imshow(digits.images[si], cmap="magma", vmin=0, vmax=16,
                                         interpolation="nearest")
                            ax_s.axis("off")
                            sc.pyplot(fig_s, use_container_width=True)
                            plt.close()

                        if proba is not None:
                            st.markdown("<br>", unsafe_allow_html=True)
                            fig_proba = px.bar(
                                x=list(range(10)), y=proba,
                                labels={"x": "Dígito", "y": "Probabilidad"},
                                color=proba, color_continuous_scale="Purples",
                                template=PLOTLY_TEMPLATE, title="Probabilidades por clase"
                            )
                            fig_proba.update_layout(
                                font_family="Space Mono", height=260,
                                coloraxis_showscale=False
                            )
                            st.plotly_chart(fig_proba, use_container_width=True)

                else:
                    st.markdown("""
                    <div style='background:#1e1e24;border:1px dashed #2a2a35;border-radius:12px;
                    padding:40px;text-align:center;color:#888899;'>
                        <div style='font-size:3rem;'>✏️</div>
                        <div style='margin-top:12px;font-family:Space Mono;font-size:0.8rem;'>
                            Dibuja un dígito y haz clic en Predecir
                        </div>
                    </div>""", unsafe_allow_html=True)

        except ImportError:
            st.error("Instala `streamlit-drawable-canvas` para activar esta función.")
            st.code("pip install streamlit-drawable-canvas")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#2a2a35;font-family:Space Mono;font-size:0.7rem;margin-top:40px;padding:20px;'>
MNIST Digit Classifier · Scikit-learn · Streamlit
</div>""", unsafe_allow_html=True)
