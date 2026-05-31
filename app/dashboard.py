import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate
from src.features import engineer, get_feature_cols
from src.data_loader import load
from src.model import train
from src.preprocessing import clean

# Constants

COMPOUND_COLORS = {
    "SOFT": "#e8002d", "MEDIUM": "#ffd600", "HARD": "#e0e0e0",
    "INTERMEDIATE": "#43b02a", "WET": "#0067ff",
}
MODEL_COLORS = {
    "XGBoost":             "#e10600",
    "Random Forest":       "#ff8800",
    "MLP":                 "#cc44ff",
    "Logistic Regression": "#4488ff",
    "Decision Tree":       "#aaaaaa",
}
BRAND = "#e10600"


def _dark_layout(title=""):
    return dict(title=title, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fafafa")


# Cached pipeline

@st.cache_data
def _load_data():
    df_raw = load()
    df, col = clean(df_raw)
    df = engineer(df, col)
    return df, col


@st.cache_resource
def _train_models():
    df, col = _load_data()
    return train(df, col)


# Page config

st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎",
    layout="wide",
)

st.markdown(
    f"<h1 style='text-align:center; color:{BRAND};'>F1 Pit Stop Strategy Predictor</h1>"
    "<p style='text-align:center; color:#888;'>"
    "DS 570 Final Project - End-to-end ML pipeline for pit stop decision prediction"
    "</p>",
    unsafe_allow_html=True,
)

# Load & train (served from cache after first run)

with st.spinner("Loading data and training models…"):
    df, col = _load_data()
    xgb, lr, rf, dt, mlp, X_train, X_test, y_train, y_test, cv_results = _train_models()

feature_cols = get_feature_cols()

all_results = {
    "Decision Tree":       evaluate(dt,  X_test, y_test),
    "Logistic Regression": evaluate(lr,  X_test, y_test),
    "Random Forest":       evaluate(rf,  X_test, y_test),
    "MLP":                 evaluate(mlp, X_test, y_test),
    "XGBoost":             evaluate(xgb, X_test, y_test),
}

target_col   = col["target"]
compound_col = col["compound"]
tyre_age_col = col["tyre_age"]
lap_time_col = col["lap_time"]
race_col     = col["race"]
year_col     = col["year"]
driver_col   = col["driver"]
position_col = col["position"]
lap_col      = col["lap"]

# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "🏁 Model Comparison",
    "🔍 XGBoost Details",
    "🏎 Race Strategy",
])


# -------------------------
# TAB 1 - Data Explorer
# -------------------------

with tab1:

    # data.head()
    st.subheader("data.head()  —  First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    # data.info() + data.isnull().sum()
    ca, cb = st.columns(2)

    info_df = pd.DataFrame({
        "Column":         df.columns,
        "Dtype":          df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count":     df.isnull().sum().values,
        "Null %":         (df.isnull().mean() * 100).round(2).values,
    })
    ca.subheader("data.info()  —  Column Types & Null Counts")
    ca.dataframe(info_df, use_container_width=True, hide_index=True)

    null_series = df.isnull().sum()
    null_df = pd.DataFrame({
        "Column":     null_series.index,
        "Null Count": null_series.values,
        "Null %":     (null_series / len(df) * 100).round(2).values,
    })
    cb.subheader("data.isnull().sum()")
    cb.dataframe(null_df, use_container_width=True, hide_index=True)

    # data.describe()
    st.subheader("data.describe() - Statistical Summary")
    st.dataframe(df.describe().round(3), use_container_width=True)

    # value_counts()
    vc1, vc2, vc3 = st.columns(3)

    vc_compound = df[compound_col].value_counts().reset_index()
    vc_compound.columns = ["Compound", "Count"]
    vc_compound["% of Laps"] = (vc_compound["Count"] / len(df) * 100).round(2)
    vc1.subheader(f"value_counts() - {compound_col}")
    vc1.dataframe(vc_compound, use_container_width=True, hide_index=True)

    vc_year = df[year_col].value_counts().sort_index().reset_index()
    vc_year.columns = [year_col, "Count"]
    vc_year["% of Laps"] = (vc_year["Count"] / len(df) * 100).round(2)
    vc2.subheader(f"value_counts() - {year_col}")
    vc2.dataframe(vc_year, use_container_width=True, hide_index=True)

    vc_target = df[target_col].value_counts().sort_index().reset_index()
    vc_target.columns = [target_col, "Count"]
    vc_target["% of Laps"] = (vc_target["Count"] / len(df) * 100).round(2)
    vc3.subheader(f"value_counts() - {target_col}")
    vc3.dataframe(vc_target, use_container_width=True, hide_index=True)

    st.divider()

    # KPI metrics
    st.subheader("Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Laps",        f"{len(df):,}")
    m2.metric("Races",             df[race_col].nunique())
    m3.metric("Drivers",           df[driver_col].nunique())
    m4.metric("Seasons",           df[year_col].nunique())
    m5.metric("Pit-Next-Lap Rate", f"{df[target_col].mean():.1%}")

    st.divider()

    # Target distribution (class imbalance)
    target_counts = df[target_col].value_counts().sort_index()
    target_pcts   = target_counts / target_counts.sum()
    fig_target = go.Figure(go.Bar(
        x=["No Pit (0)", "Pit Next Lap (1)"],
        y=target_counts.values,
        marker_color=["#4488ff", BRAND],
        text=[f"{c:,} ({p:.1%})" for c, p in zip(target_counts.values, target_pcts.values)],
        textposition="outside",
    ))
    fig_target.update_layout(
        yaxis_title="Number of Laps",
        **_dark_layout("Target Variable Distribution - PitNextLap (Class Imbalance)"),
    )
    st.plotly_chart(fig_target, use_container_width=True)

    # Compound & Yearly pit rates
    ca, cb = st.columns(2)

    rates = (df.groupby(compound_col)[target_col].mean().reset_index()
               .rename(columns={target_col: "pit_rate"})
               .sort_values("pit_rate", ascending=False))
    fig_compound = go.Figure(go.Bar(
        x=rates[compound_col], y=rates["pit_rate"],
        marker_color=[COMPOUND_COLORS.get(c.upper(), "#888") for c in rates[compound_col]],
        text=rates["pit_rate"].map("{:.1%}".format), textposition="outside",
    ))
    fig_compound.update_layout(yaxis_tickformat=".0%",
                                **_dark_layout("Pit Rate by Tyre Compound"))
    ca.plotly_chart(fig_compound, use_container_width=True)

    yearly = (df.groupby(year_col)[target_col].mean().reset_index()
                .rename(columns={target_col: "pit_rate"}))
    fig_yearly = go.Figure(go.Bar(
        x=yearly[year_col].astype(str), y=yearly["pit_rate"],
        marker_color=BRAND,
        text=yearly["pit_rate"].map("{:.1%}".format), textposition="outside",
    ))
    fig_yearly.update_layout(yaxis_tickformat=".0%",
                              **_dark_layout("Pit Rate by Season"))
    cb.plotly_chart(fig_yearly, use_container_width=True)

    # Tyre age line & histogram 
    ca, cb = st.columns(2)

    age_rates = (df.groupby(tyre_age_col)[target_col].mean().reset_index()
                   .rename(columns={target_col: "pit_rate"}))
    fig_age = px.line(age_rates, x=tyre_age_col, y="pit_rate",
                      labels={"pit_rate": "Pit Rate", tyre_age_col: "Tyre Age (laps)"})
    fig_age.update_traces(line_color=BRAND)
    fig_age.update_layout(yaxis_tickformat=".0%",
                           **_dark_layout("Pit Rate by Tyre Age"))
    ca.plotly_chart(fig_age, use_container_width=True)

    fig_hist = go.Figure()
    for label, val, color in [("No Pit (0)", 0, "#4488ff"), ("Pit Next Lap (1)", 1, BRAND)]:
        vals = df[df[target_col] == val][tyre_age_col].dropna()
        fig_hist.add_trace(go.Histogram(
            x=vals, name=label, marker_color=color,
            opacity=0.7, nbinsx=40, histnorm="percent",
        ))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Tyre Life (laps)", yaxis_title="% of laps",
        **_dark_layout("Tyre Life Distribution: Pit vs No Pit"),
    )
    cb.plotly_chart(fig_hist, use_container_width=True)

    #  Lap time box & position pit rate
    ca, cb = st.columns(2)

    fig_lt = go.Figure()
    for compound, grp in df.groupby(compound_col):
        vals = grp[lap_time_col].dropna()
        vals = vals[vals.between(vals.quantile(0.01), vals.quantile(0.99))]
        fig_lt.add_trace(go.Box(
            y=vals, name=str(compound),
            marker_color=COMPOUND_COLORS.get(str(compound).upper(), "#888"),
        ))
    fig_lt.update_layout(**_dark_layout("Lap Time by Compound (seconds)"))
    ca.plotly_chart(fig_lt, use_container_width=True)

    pos_rates = (df[df[position_col].between(1, 20)]
                   .groupby(position_col)[target_col].mean().reset_index()
                   .rename(columns={target_col: "pit_rate"}))
    fig_pos = go.Figure(go.Bar(
        x=pos_rates[position_col].astype(str), y=pos_rates["pit_rate"],
        marker_color=BRAND,
        text=pos_rates["pit_rate"].map("{:.1%}".format), textposition="outside",
    ))
    fig_pos.update_layout(
        xaxis_title="Race Position", yaxis_tickformat=".0%",
        **_dark_layout("Pit Rate by Race Position"),
    )
    cb.plotly_chart(fig_pos, use_container_width=True)

    # Driver pit rate (top 20) 
    driver_rates = (df.groupby(driver_col)[target_col]
                      .agg(pit_rate="mean", laps="count").reset_index()
                      .query("laps >= 100")
                      .sort_values("pit_rate", ascending=True)
                      .tail(20))
    fig_driver = go.Figure(go.Bar(
        x=driver_rates["pit_rate"], y=driver_rates[driver_col],
        orientation="h", marker_color=BRAND,
        text=driver_rates["pit_rate"].map("{:.1%}".format), textposition="outside",
    ))
    fig_driver.update_layout(
        xaxis_tickformat=".0%", height=500,
        **_dark_layout("Pit Rate by Driver (top 20, min 100 laps)"),
    )
    st.plotly_chart(fig_driver, use_container_width=True)

    # Correlation heatmap
    num_cols = ["TyreLife", "LapNumber", "LapTimeSec", "Position", "Stint",
                "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
                "Normalized_TyreLife", "Position_Change", "PitStop", target_col]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Feature Correlation Matrix")
    fig_corr.update_layout(**_dark_layout(), height=550)
    st.plotly_chart(fig_corr, use_container_width=True)


# ----------------------------
# TAB 2 - Model Comparison
# ----------------------------

with tab2:
    st.subheader("Model Performance Comparison")

    rows = []
    for name, res in all_results.items():
        cv = cv_results.get(name, {})
        cv_text = (f"{cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}"
                   if cv.get("cv_mean") is not None else "—")
        rows.append({
            "Model":                   name,
            "ROC-AUC":                 res["roc_auc"],
            "Accuracy":                res["accuracy"],
            "F1":                      res["f1"],
            "Precision":               res["precision"],
            "Recall":                  res["recall"],
            "CV ROC-AUC (mean ± std)": cv_text,
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["ROC-AUC", "Accuracy", "F1", "Precision", "Recall"],
            color="#1a3a1a",
            axis=0,
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "CV uses TimeSeriesSplit (5 folds) — each fold's test window is strictly "
        "after its train window, preserving chronological order."
    )

    ca, cb = st.columns(2)

    fig_roc = go.Figure()
    for name, res in all_results.items():
        fig_roc.add_trace(go.Scatter(
            x=res["roc_fpr"], y=res["roc_tpr"],
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            line=dict(color=MODEL_COLORS.get(name, "#888"), width=2),
        ))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                  line=dict(color="#555", dash="dash")))
    fig_roc.update_layout(xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate",
                           **_dark_layout("ROC Curves - All Models"))
    ca.plotly_chart(fig_roc, use_container_width=True)

    fig_pr = go.Figure()
    for name, res in all_results.items():
        fig_pr.add_trace(go.Scatter(
            x=res["pr_recall"], y=res["pr_precision"],
            name=name,
            line=dict(color=MODEL_COLORS.get(name, "#888"), width=2),
        ))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          **_dark_layout("Precision-Recall Curves — All Models"))
    cb.plotly_chart(fig_pr, use_container_width=True)


# ---------------------------- 
# TAB 3 - XGBoost Details
# ----------------------------

with tab3:
    xgb_res = all_results["XGBoost"]

    ca, cb = st.columns(2)

    cm = xgb_res["confusion_matrix"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted: No Pit", "Predicted: Pit"],
        y=["Actual: No Pit", "Actual: Pit"],
        color_continuous_scale="Reds",
        title="Confusion Matrix — XGBoost",
    )
    fig_cm.update_layout(**_dark_layout()) 
    ca.plotly_chart(fig_cm, use_container_width=True)

    fi_df = (pd.DataFrame({"feature": feature_cols, "importance": xgb.feature_importances_})
               .sort_values("importance", ascending=True))
    fig_fi = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h", marker_color=BRAND,
    ))
    fig_fi.update_layout(height=max(300, len(feature_cols) * 32),
                          **_dark_layout("XGBoost Feature Importance"))
    cb.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Classification Report — XGBoost")
    st.code(xgb_res["classification_report"], language=None)

# ----------------------------
# TAB 4 - Race Strategy
# ----------------------------

with tab4:
    st.subheader("Race Strategy - Pit-Stop Probability per Lap")

    races = sorted(df[race_col].dropna().unique())
    ca, cb = st.columns(2)
    selected_race   = ca.selectbox("Select Race", races, index=len(races) - 1)
    race_drivers    = sorted(
        df[df[race_col].astype(str) == str(selected_race)][driver_col].dropna().unique()
    )
    selected_driver = cb.selectbox("Select Driver", race_drivers)

    mask = ((df[race_col].astype(str) == str(selected_race)) &
            (df[driver_col].astype(str) == str(selected_driver)))
    sub  = df[mask].copy().sort_values(lap_col)

    if not sub.empty:
        X_sub = sub[[c for c in feature_cols if c in sub.columns]].copy()
        for fc in feature_cols:
            if fc not in X_sub.columns:
                X_sub[fc] = 0
        X_sub  = X_sub[feature_cols].fillna(0)
        probs  = xgb.predict_proba(X_sub)[:, 1]
        actual = sub[target_col].values
        laps   = sub[lap_col].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=laps, y=probs, mode="lines", name="Pit Probability",
            line=dict(color=BRAND, width=2),
        ))
        pit_laps = laps[actual == 1]
        if len(pit_laps):
            fig.add_trace(go.Scatter(
                x=pit_laps, y=probs[actual == 1], mode="markers",
                name="Actual Pit (Next Lap)",
                marker=dict(color="#ffd600", size=12, symbol="diamond"),
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#555",
                      annotation_text="50% threshold")
        fig.update_layout(
            xaxis_title="Lap", yaxis_title="Probability", yaxis_range=[0, 1],
            **_dark_layout(f"Pit-Stop Probability - {selected_driver} @ {selected_race}"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data for the selected race / driver combination.")
