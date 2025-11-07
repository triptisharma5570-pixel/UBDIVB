import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(layout="wide", page_title="Universal Bank - Personal Loan Analytics")

st.title("Universal Bank — Personal Loan Prediction & Marketing Insights")
st.markdown("Streamlit dashboard to explore the Universal Bank dataset, train models (Decision Tree, Random Forest, Gradient Boosting), and predict Personal Loan acceptance.")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# --- Utilities ---
def preprocess(df):
    # Ensure expected columns exist, and return X, y
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    y = None
    if "Personal Loan" in df.columns:
        y = df["Personal Loan"]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df, y

def train_models(X_train, y_train, n_estimators=100, random_state=42):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models

def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        probs = model.decision_function(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, probs),
        "y_pred": y_pred,
        "probs": probs
    }

def create_download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# --- Load sample by default ---
st.sidebar.header("Data / Settings")
use_sample = st.sidebar.checkbox("Load sample dataset (UniversalBank.csv) in this package", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload your CSV dataset", type=["csv"])

if use_sample:
    if not st.sidebar.button("Reload sample"):
        pass
    data = load_data("UniversalBank.csv")
elif uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.info("Choose to load the sample dataset or upload your own CSV from the sidebar.")
    st.stop()

df_raw, y_series = preprocess(data)
st.sidebar.write("Dataset shape:", df_raw.shape)

# Tabs for app
tabs = st.tabs(["EDA & Insights", "Train & Evaluate Models", "Predict & Download"])

# ---------- EDA & Insights ----------
with tabs[0]:
    st.header("Exploratory Data Analysis — Actionable Marketing Insights")
    st.markdown("Five interactive charts with complex insights to help you prioritize customer segments for conversion campaigns.")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("1) Income distribution by Education & Personal Loan (stacked)")
        if "Income" in df_raw.columns:
            df_plot = df_raw.copy()
            df_plot["Education_label"] = df_plot["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
            fig = px.histogram(df_plot, x="Income", color="Education_label", facet_row=None, nbins=40, barmode="overlay",
                               hover_data=["Personal Loan"], title="Income distribution by Education level")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Action:** Target high-income graduates/advanaced segments where acceptance rate is high.")

    with col2:
        st.subheader("2) Conversion lift by Income deciles (decile vs conversion rate)")
        if "Income" in df_raw.columns and "Personal Loan" in data.columns:
            df_lift = df_raw.copy()
            df_lift["Income_decile"] = pd.qcut(df_lift["Income"], 10, labels=False, duplicates='drop')
            lift = df_lift.groupby("Income_decile")["Personal Loan"].agg(["count","mean"]).reset_index().rename(columns={"mean":"conversion_rate"})
            lift["decile_label"] = lift["Income_decile"].astype(int)+1
            fig2 = px.bar(lift, x="decile_label", y="conversion_rate", text="count", title="Conversion rate by Income decile (higher decile = richer)")
            fig2.update_layout(xaxis_title="Income decile (1 = lowest)", yaxis_title="Conversion rate")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("**Action:** Focus marketing budget on deciles with high conversion but manageable size (count).")

    st.markdown("---")
    st.subheader("3) CCAvg spending vs Income colored by Personal Loan acceptance (segmentation view)")
    if {"CCAvg","Income","Personal Loan"}.issubset(df_raw.columns):
        fig3 = px.scatter(df_raw, x="Income", y="CCAvg", color="Personal Loan", size="Family", hover_data=["Age","Education"], title="CCAvg vs Income (size = family)")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**Action:** Identify customers with moderate income but high CCAvg — good cross-sell candidates.")

    st.markdown("---")
    st.subheader("4) Acceptance rate heatmap: Education vs Family size")
    if {"Education","Family","Personal Loan"}.issubset(df_raw.columns):
        pivot = df_raw.pivot_table(index="Education", columns="Family", values="Personal Loan", aggfunc="mean")
        pivot.index = pivot.index.map({1:"Undergrad",2:"Graduate",3:"Advanced"})
        fig4 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index, colorbar_title="Acceptance rate"))
        fig4.update_layout(title="Acceptance rate by Education level and Family size", xaxis_title="Family size", yaxis_title="Education level")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("**Action:** Tailor messaging by family size (e.g., family loan bundles for larger families).")

    st.markdown("---")
    st.subheader("5) Experience vs Income with rolling acceptance rate (trend)")
    if {"Experience","Income","Personal Loan"}.issubset(df_raw.columns):
        df_trend = df_raw.copy()
        df_trend = df_trend.sort_values("Experience")
        df_trend["rolling_accept_rate"] = df_trend["Personal Loan"].rolling(window=50, min_periods=10).mean()
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df_trend["Experience"], y=df_trend["rolling_accept_rate"], mode="lines", name="Rolling acceptance rate"))
        fig5.add_trace(go.Scatter(x=df_trend["Experience"], y=df_trend["Income"]/df_trend["Income"].max(), mode="markers", name="Normalized Income (points)", opacity=0.4))
        fig5.update_layout(title="Rolling acceptance rate by Experience (with Income overlay)", xaxis_title="Experience (years)", yaxis_title="Rolling acceptance rate (and normalized income)")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("**Action:** Use experience cohorts to tailor eligibility and messaging.")

# ---------- Model Training & Evaluation ----------
with tabs[1]:
    st.header("Train models & evaluate performance")
    st.markdown("Click **Train all models** to train Decision Tree, Random Forest and Gradient Boosting on the current dataset (Personal Loan label is required).")
    if "Personal Loan" not in data.columns:
        st.warning("Personal Loan column not found in dataset. This tab requires the label to train models.")
    else:
        test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.5, value=0.30, step=0.05)
        n_estimators = st.number_input("n_estimators for ensembles (Random Forest & GBM)", min_value=10, max_value=500, value=100, step=10)
        train_button = st.button("Train all models")
        if train_button:
            # Prepare features and target
            df_model = data.copy()
            if "ID" in df_model.columns: df_model = df_model.drop(columns=["ID"])
            X = df_model.drop(columns=["Personal Loan"])
            y = df_model["Personal Loan"]
            # Ensure numeric
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.factorize(X[col])[0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            models = train_models(X_train, y_train, n_estimators=int(n_estimators))
            results = {}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for name, model in models.items():
                train_m = compute_metrics(model, X_train, y_train)
                test_m = compute_metrics(model, X_test, y_test)
                cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
                fi = getattr(model, "feature_importances_", None)
                results[name] = {"train": train_m, "test": test_m, "cv_mean": cv_auc.mean(), "cv_std": cv_auc.std(), "model": model, "feature_importance": fi}
            # Summary table
            summary = []
            for name, r in results.items():
                summary.append({
                    "Algorithm": name,
                    "Train Acc": round(r["train"]["accuracy"],4),
                    "Test Acc": round(r["test"]["accuracy"],4),
                    "Precision": round(r["test"]["precision"],4),
                    "Recall": round(r["test"]["recall"],4),
                    "F1": round(r["test"]["f1"],4),
                    "AUC (test)": round(r["test"]["auc"],4),
                    "CV AUC (5-fold mean)": f'{r["cv_mean"]:.4f} ± {r["cv_std"]:.4f}'
                })
            st.subheader("Performance summary")
            st.table(pd.DataFrame(summary).set_index("Algorithm"))
            # ROC combined plot
            fig = go.Figure()
            for name, r in results.items():
                fpr, tpr, _ = roc_curve(y_test, r["test"]["probs"])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['test']['auc']:.3f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash='dash'), name="Random"))
            fig.update_layout(title="ROC curves (Test set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
            # Confusion matrices and feature importances
            st.subheader("Confusion matrices (Test)")
            cm_cols = st.columns(3)
            for i, (name, r) in enumerate(results.items()):
                with cm_cols[i]:
                    cm = confusion_matrix(y_test, r["test"]["y_pred"])
                    st.text(name)
                    st.write(cm)
            st.subheader("Top feature importances (per model)")
            for name, r in results.items():
                fi = r["feature_importance"]
                if fi is not None:
                    feat_imp = pd.DataFrame({"feature": X.columns, "importance": fi}).sort_values("importance", ascending=False).head(10)
                    st.write(name)
                    st.bar_chart(feat_imp.set_index("feature")["importance"])
            # Persist models to session state
            st.session_state["trained_models"] = results
            st.success("Training complete. Models saved in session state.")

# ---------- Predict & Download ----------
with tabs[2]:
    st.header("Predict on new data & download labelled file")
    st.markdown("Upload a CSV (it must contain the same features as the training dataset). Pick which model to use for prediction. Download the CSV with predicted `Personal Loan` label.")
    uploaded = st.file_uploader("Upload CSV to predict", type=["csv"])
    chosen_model = st.selectbox("Choose trained model (session)", options=["Decision Tree","Random Forest","Gradient Boosting"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        # Preprocess similarly
        new_df_proc, _ = preprocess(new_df)
        if "Personal Loan" in new_df_proc.columns:
            new_df_proc = new_df_proc.drop(columns=["Personal Loan"])
        for col in new_df_proc.select_dtypes(include=['object']).columns:
            new_df_proc[col] = pd.factorize(new_df_proc[col])[0]
        if "trained_models" not in st.session_state:
            st.warning("No trained models in session. Please go to 'Train & Evaluate Models' tab and click Train all models first (models are stored only in session).")
        else:
            models = st.session_state["trained_models"]
            if chosen_model not in models:
                st.error("Chosen model not found in session. Re-train models.")
            else:
                model = models[chosen_model]["model"]
                train_cols = getattr(model, "feature_names_in_", None)
                if train_cols is not None:
                    missing = [c for c in train_cols if c not in new_df_proc.columns]
                    for c in missing:
                        new_df_proc[c] = 0
                    Xnew = new_df_proc[train_cols]
                else:
                    Xnew = new_df_proc.reindex(columns=new_df_proc.columns, fill_value=0)
                preds = model.predict(Xnew)
                new_df_out = new_df.copy()
                new_df_out["Predicted Personal Loan"] = preds
                st.write(new_df_out.head(10))
                href = create_download_link(new_df_out, filename="predicted_personal_loan.csv")
                st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("Built for marketing heads to quickly get insights and run models. You can clone this repository to Streamlit Cloud and deploy `app.py`.")

