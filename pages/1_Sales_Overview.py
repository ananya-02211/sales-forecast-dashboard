import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Overview", layout="wide")

st.title("📈 Sales Overview Dashboard")

# ----------------------------
# Load Dataset
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month_Number"] = df["Order Date"].dt.month

    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------

st.sidebar.header("Filters")

selected_region = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

selected_category = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

filtered_df = df[
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category))
]

# ----------------------------
# KPI Cards
# ----------------------------

total_sales = filtered_df["Sales"].sum()

total_orders = filtered_df["Order ID"].nunique()

total_customers = filtered_df["Customer ID"].nunique()

avg_order = filtered_df.groupby("Order ID")["Sales"].sum().mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")

col2.metric("📦 Orders", total_orders)

col3.metric("👥 Customers", total_customers)

col4.metric("🛒 Avg Order Value", f"${avg_order:,.2f}")

st.markdown("---")

# ----------------------------
# Total Sales by Year
# ----------------------------

year_sales = (
    filtered_df
    .groupby("Year")["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    year_sales,
    x="Year",
    y="Sales",
    title="Total Sales by Year",
    text_auto=".2s"
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Monthly Sales Trend
# ----------------------------

monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Region-wise Sales
# ----------------------------

col1, col2 = st.columns(2)

region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    title="Sales by Region"
)

col1.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Category Sales
# ----------------------------

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig4 = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Sales by Category"
)

col2.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# Data Table
# ----------------------------

st.markdown("---")

st.subheader("Filtered Sales Data")

st.dataframe(filtered_df)