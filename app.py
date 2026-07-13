import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting and Business Analytics Dashboard")

st.markdown("""
### Welcome!

This dashboard was developed as part of the internship project.

Use the navigation panel on the left to explore:

- 📈 Sales Overview Dashboard
- 🔮 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments

---
""")

st.info("Select a page from the left sidebar to begin exploring the dashboard.")