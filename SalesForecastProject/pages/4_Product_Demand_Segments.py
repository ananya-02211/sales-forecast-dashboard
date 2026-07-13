import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(page_title="Product Demand Segments", layout="wide")

st.title("📦 Product Demand Segments")

st.markdown("""
This page groups product sub-categories into demand segments using the
K-Means clustering algorithm.
""")

# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month

    return df

df = load_data()

# --------------------------------------------------
# Monthly Sales
# --------------------------------------------------

monthly_sales = df.groupby(
    ["Sub-Category","Year","Month"]
)["Sales"].sum().reset_index()

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

total_sales = monthly_sales.groupby("Sub-Category")["Sales"].sum()

volatility = monthly_sales.groupby("Sub-Category")["Sales"].std()

average_order = df.groupby("Sub-Category")["Sales"].mean()

yearly_sales = df.groupby(
    ["Sub-Category","Year"]
)["Sales"].sum().reset_index()

growth = yearly_sales.groupby("Sub-Category")["Sales"].pct_change()

growth = growth.groupby(yearly_sales["Sub-Category"]).mean()

cluster_df = pd.DataFrame({

    "TotalSales": total_sales,

    "Volatility": volatility,

    "AverageOrderValue": average_order,

    "GrowthRate": growth

})

cluster_df = cluster_df.fillna(0)

# --------------------------------------------------
# Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(cluster_df)

# --------------------------------------------------
# KMeans
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_df["Cluster"] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# Cluster Names
# --------------------------------------------------

cluster_names = {

    0:"High Volume, Stable Demand",

    1:"Growing Demand",

    2:"Low Volume, High Volatility",

    3:"Declining Demand"

}

cluster_df["Demand Segment"] = cluster_df["Cluster"].map(cluster_names)

# --------------------------------------------------
# PCA
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

cluster_df["PCA1"] = X_pca[:,0]

cluster_df["PCA2"] = X_pca[:,1]

cluster_df.reset_index(inplace=True)

# --------------------------------------------------
# Scatter Plot
# --------------------------------------------------

fig = px.scatter(

    cluster_df,

    x="PCA1",

    y="PCA2",

    color="Demand Segment",

    hover_name="Sub-Category",

    size="TotalSales",

    title="Product Demand Segments"

)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Cluster Table
# --------------------------------------------------

st.subheader("Sub-Category Demand Segments")

st.dataframe(

    cluster_df[
        [
            "Sub-Category",
            "Demand Segment",
            "TotalSales",
            "GrowthRate",
            "AverageOrderValue"
        ]
    ]

)

# --------------------------------------------------
# Summary
# --------------------------------------------------

st.subheader("Cluster Distribution")

summary = cluster_df.groupby("Demand Segment").size().reset_index(name="Count")

fig2 = px.bar(

    summary,

    x="Demand Segment",

    y="Count",

    color="Demand Segment",

    title="Number of Sub-Categories in Each Segment"

)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Business Recommendations
# --------------------------------------------------

st.subheader("Business Recommendations")

st.markdown("""

### 📈 High Volume, Stable Demand
- Maintain high inventory levels.
- Ensure continuous stock availability.
- Prioritize these products.

### 🚀 Growing Demand
- Increase inventory gradually.
- Launch promotional campaigns.
- Monitor demand growth regularly.

### ⚠ Low Volume, High Volatility
- Stock carefully.
- Review frequently.
- Avoid overstocking.

### 📉 Declining Demand
- Reduce inventory.
- Offer discounts.
- Consider product replacement.

""")