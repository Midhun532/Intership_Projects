import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")

# Caching
@st.cache_data
def load_data():
    return pd.read_csv("winequality.csv")

@st.cache_resource
def load_model():
    return joblib.load("wine_rf_model.joblib")

# Load assets
df = load_data()
model = load_model()

# App layout
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("food-wine-wallpaper.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .reportview-container .main .block-container{{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Wine Quality Analyzer")
tabs = st.tabs(["Wine Quality Prediction", "EDA", "Feature Importance", "Outlier Detection"])

# Sidebar
with st.sidebar:
    st.header("Enter Wine Properties")
    features = {}
    for col in df.columns[:-1]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        step = (max_val - min_val) / 100
        features[col] = st.slider(col, min_val, max_val, float(df[col].mean()), step=step)
    input_df = pd.DataFrame([features])

# Prediction
with tabs[0]:
    st.subheader("Prediction Result")
    prediction = model.predict(input_df)[0]

    # Category Label
    if prediction <= 4:
        label = "Poor Quality"
        color = "red"
    elif prediction <= 6:
        label = "Average Quality"
        color = "orange"
    else:
        label = "Good Quality"
        color = "green"

    st.metric("Predicted Quality Score", f"{prediction:.2f}")
    st.success(f"Label: {label} ({prediction:.2f})")

    # Outlier Detection
    iqr_data = df[['alcohol', 'volatile acidity']]
    outliers = {}
    for col in iqr_data.columns:
        Q1 = iqr_data[col].quantile(0.25)
        Q3 = iqr_data[col].quantile(0.75)
        IQR = Q3 - Q1
        if not (Q1 - 1.5 * IQR <= features[col] <= Q3 + 1.5 * IQR):
            outliers[col] = (features[col], Q1, Q3)

    if outliers:
        st.warning("⚠️ Outlier(s) Detected:")
        for col, (val, Q1, Q3) in outliers.items():
            st.write(f"{col}: {val:.2f} (Expected range: {Q1:.2f}–{Q3:.2f})")
    else:
        st.info("All input values are within safe ranges.")

# EDA tab
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='quality', title='Wine Quality Distribution')
        st.plotly_chart(fig)

        fig2 = px.scatter(df, x='alcohol', y='quality', trendline='ols', title='Alcohol vs. Quality')
        st.plotly_chart(fig2)

    with col2:
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig_corr)

        fig_box = px.box(df, x='quality', y='volatile acidity', title='Volatile Acidity by Quality')
        st.plotly_chart(fig_box)

# Feature importance
with tabs[2]:
    st.subheader("Top 10 Feature Importances")
    X_cols = df.drop(columns=['quality']).columns
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'feature': X_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top10 = feat_df.head(10)
    fig = go.Figure(go.Bar(
        x=top10['importance'][::-1],
        y=top10['feature'][::-1],
        orientation='h',
        marker_color='dodgerblue'
    ))
    fig.update_layout(title='Top 10 Important Features')
    st.plotly_chart(fig)
    st.dataframe(feat_df)

# Outlier Detection tab
with tabs[3]:
    st.subheader("Outlier Detection")

    iso = IsolationForest()
    preds = iso.fit_predict(df[['alcohol', 'volatile acidity']])
    df['Outlier'] = np.where(preds == -1, 'Yes', 'No')
    outlier_records = df[df['Outlier'] == 'Yes']

    st.dataframe(outlier_records[['alcohol', 'volatile acidity', 'Outlier']].reset_index(drop=True))
