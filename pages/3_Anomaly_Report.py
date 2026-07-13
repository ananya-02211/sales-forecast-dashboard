import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(page_title="Anomaly Report", layout="wide")

st.title("🚨 Anomaly Detection Report")

st.markdown("""
This page identifies unusual sales records using the **Isolation Forest**
algorithm. The detected anomalies may represent exceptionally high or
low sales values that require further business investigation.
""")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("anomaly_detection.csv")

    return df

df = load_data()

# ---------------------------------------------------
# Isolation Forest
# ---------------------------------------------------

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = model.fit_predict(df[["Global_Sales"]])

anomaly_df = df[df["Anomaly"] == -1]

# ---------------------------------------------------
# Scatter Plot
# ---------------------------------------------------

fig = px.scatter(
    df,
    x="Year",
    y="Global_Sales",
    color=df["Anomaly"].astype(str),
    color_discrete_map={
        "1": "blue",
        "-1": "red"
    },
    hover_data=["Name", "Platform", "Genre"],
    title="Isolation Forest Anomaly Detection"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

col1, col2 = st.columns(2)

col1.metric(
    "Total Records",
    len(df)
)

col2.metric(
    "Detected Anomalies",
    len(anomaly_df)
)

# ---------------------------------------------------
# Table
# ---------------------------------------------------

st.subheader("Detected Anomalies")

st.dataframe(
    anomaly_df[
        [
            "Name",
            "Year",
            "Platform",
            "Genre",
            "Global_Sales"
        ]
    ]
)

# ---------------------------------------------------
# Business Insights
# ---------------------------------------------------

st.subheader("Business Interpretation")

st.markdown("""

### Key Findings

- Isolation Forest detected unusual sales observations.

- These records represent exceptionally high or low global sales.

- Such anomalies may be caused by blockbuster game releases,
  promotional events, market disruptions or data quality issues.

- Investigating these anomalies helps businesses understand
  exceptional market behaviour and improve forecasting accuracy.

""")