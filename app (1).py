import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Propensity Score Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = [
    "customerID",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TechSupport",
    "PaymentMethod",
    "Churn",
]

RISK_WEIGHTS = {
    "Contract_One year": -0.3,
    "Contract_Two year": -0.6,
    "TechSupport_No_Service": 0.4,
    "PaymentMethod_Credit card (automatic)": -0.1,
    "PaymentMethod_Electronic check": 0.3,
    "PaymentMethod_Mailed check": 0.1,
    "tenure_scaled_inverse": 0.5,
    "MonthlyCharges_scaled": 0.2,
    "TotalCharges_scaled_inverse": 0.1,
}

RISK_COLORS = {
    "Very High": "#DC2626",
    "High": "#F97316",
    "Moderate": "#CA8A04",
    "Low": "#16A34A",
}


def validate_columns(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    df["TechSupport"] = df["TechSupport"].replace({"No internet service": "No"})
    df["Churn_numeric"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.dropna(subset=["TotalCharges", "MonthlyCharges", "tenure"])
    df = df.reset_index(drop=True)
    return df



def min_max_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    min_val, max_val = series.min(), series.max()
    if pd.isna(min_val) or pd.isna(max_val):
        result = pd.Series([50.0] * len(series), index=series.index)
    elif max_val == min_val:
        result = pd.Series([50.0] * len(series), index=series.index)
    else:
        result = ((series - min_val) / (max_val - min_val) * 100).round(2)
    return 100 - result if invert else result



def classify_risk(score: float) -> str:
    if score >= 75:
        return "Very High"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Moderate"
    return "Low"



def compute_churn_propensity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    working = df.copy()

    encoded = pd.get_dummies(
        working[[
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "TechSupport",
            "PaymentMethod",
        ]],
        columns=["Contract", "TechSupport", "PaymentMethod"],
        drop_first=False,
    )

    encoded = encoded.rename(columns={"TechSupport_No": "TechSupport_No_Service"})
    encoded = encoded.drop(columns=["TechSupport_Yes"], errors="ignore")
    encoded = encoded.drop(columns=["Contract_Month-to-month"], errors="ignore")
    encoded = encoded.drop(columns=["PaymentMethod_Bank transfer (automatic)"], errors="ignore")

    encoded["tenure_scaled_inverse"] = min_max_normalize(encoded["tenure"], invert=True)
    encoded["MonthlyCharges_scaled"] = min_max_normalize(encoded["MonthlyCharges"])
    encoded["TotalCharges_scaled_inverse"] = min_max_normalize(encoded["TotalCharges"], invert=True)

    encoded["raw_risk_score"] = 0.0
    contribution_means = {}

    for feature, weight in RISK_WEIGHTS.items():
        if feature not in encoded.columns:
            encoded[feature] = 0
        contribution = encoded[feature] * weight
        encoded["raw_risk_score"] += contribution
        contribution_means[feature] = contribution.abs().mean()

    encoded["churn_risk_score"] = min_max_normalize(encoded["raw_risk_score"])
    encoded["risk_band"] = encoded["churn_risk_score"].apply(classify_risk)

    result = working.merge(
        encoded[["customerID", "churn_risk_score", "risk_band"]],
        on="customerID",
        how="left",
    )

    result = result.sort_values("churn_risk_score", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    contribution_series = pd.Series(contribution_means).sort_values(ascending=False)
    return result, contribution_series



def chart_histogram(result_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        result_df,
        x="churn_risk_score",
        nbins=30,
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        title="Churn Risk Distribution",
        labels={"churn_risk_score": "Churn Risk Score (0-100)", "count": "Customers"},
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05, title_font_size=16, legend_title_text="Risk Band")
    return fig



def chart_top50(top50_df: pd.DataFrame) -> go.Figure:
    plot_df = top50_df.sort_values("churn_risk_score", ascending=True)
    fig = px.bar(
        plot_df,
        x="churn_risk_score",
        y="customerID",
        orientation="h",
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        title="Top 50 At-Risk Customers",
        labels={"customerID": "Customer ID", "churn_risk_score": "Churn Risk Score (0-100)"},
        template="plotly_white",
        hover_data=["Contract", "tenure", "MonthlyCharges", "Churn"],
    )
    fig.update_layout(height=max(700, len(top50_df) * 18), title_font_size=16, legend_title_text="Risk Band")
    return fig



def chart_contributions(contribution_series: pd.Series) -> go.Figure:
    labels_map = {
        "tenure_scaled_inverse": "Low tenure",
        "MonthlyCharges_scaled": "High monthly charges",
        "TotalCharges_scaled_inverse": "Low total charges",
        "TechSupport_No_Service": "No tech support",
        "Contract_Two year": "Two-year contract",
        "Contract_One year": "One-year contract",
        "PaymentMethod_Electronic check": "Electronic check",
        "PaymentMethod_Mailed check": "Mailed check",
        "PaymentMethod_Credit card (automatic)": "Credit card (automatic)",
    }
    plot_df = contribution_series.reset_index()
    plot_df.columns = ["feature", "avg_abs_contribution"]
    plot_df["feature"] = plot_df["feature"].map(lambda x: labels_map.get(x, x))

    fig = px.bar(
        plot_df,
        x="avg_abs_contribution",
        y="feature",
        orientation="h",
        title="Risk Factor Contribution",
        labels={
            "feature": "Risk Factor",
            "avg_abs_contribution": "Average Absolute Weighted Contribution",
        },
        template="plotly_white",
    )
    fig.update_layout(height=450, title_font_size=16)
    return fig


with st.sidebar:
    st.title("Churn Propensity Analyzer")
    st.caption("Subscription & SaaS BI Tool")
    st.divider()
    st.markdown("### How to use")
    st.markdown(
        "1. Upload the cleaned telco churn CSV  \n"
        "2. Apply filters to focus on a segment  \n"
        "3. Review KPI cards and charts  \n"
        "4. Download the scored customer watch-list"
    )
    st.divider()
    st.markdown("### Required columns")
    for col in REQUIRED_COLUMNS:
        st.markdown(f"- `{col}`")

st.title("Churn Propensity Score Dashboard")
st.markdown(
    "Upload your Telco Customer Churn dataset to score each customer's churn risk, "
    "identify the highest-risk accounts, and review the strongest churn signals."
)

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    sample = pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "tenure": [1, 34, 2],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": [29.85, 1889.50, 108.15],
            "TechSupport": ["No", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
            "Churn": ["No", "No", "Yes"],
        }
    )
    st.dataframe(sample, use_container_width=True)
    st.stop()

raw_df = pd.read_csv(uploaded_file)
missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(
        f"Missing required columns: {missing_cols}. Your file has: {list(raw_df.columns)}"
    )
    st.stop()

df_clean = clean_data(raw_df)

with st.expander("Data Preview", expanded=False):
    st.dataframe(df_clean.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df_clean):,}")
    c2.metric("Churned Customers", f"{(df_clean['Churn'] == 'Yes').sum():,}")
    c3.metric("Average Monthly Charges", f"${df_clean['MonthlyCharges'].mean():.2f}")

st.divider()
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

with col1:
    contract_options = ["All Contracts"] + sorted(df_clean["Contract"].dropna().unique().tolist())
    contract_choice = st.selectbox("Contract Type", contract_options)

with col2:
    payment_options = ["All Payment Methods"] + sorted(df_clean["PaymentMethod"].dropna().unique().tolist())
    payment_choice = st.selectbox("Payment Method", payment_options)

with col3:
    churn_options = ["All Customers", "Churned Only", "Active Only"]
    churn_choice = st.selectbox("Customer Status", churn_options)

filtered_df = df_clean.copy()
if contract_choice != "All Contracts":
    filtered_df = filtered_df[filtered_df["Contract"] == contract_choice]
if payment_choice != "All Payment Methods":
    filtered_df = filtered_df[filtered_df["PaymentMethod"] == payment_choice]
if churn_choice == "Churned Only":
    filtered_df = filtered_df[filtered_df["Churn"] == "Yes"]
elif churn_choice == "Active Only":
    filtered_df = filtered_df[filtered_df["Churn"] == "No"]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

scored_df, contribution_series = compute_churn_propensity(filtered_df)
top50_df = scored_df.head(50).copy()

st.divider()
st.subheader("Headline Metrics")
m1, m2, m3, m4 = st.columns(4)

at_risk_count = (scored_df["churn_risk_score"] >= 75).sum()
monthly_revenue_at_risk = top50_df["MonthlyCharges"].sum()
avg_risk_score = scored_df["churn_risk_score"].mean()
top_customer = top50_df.iloc[0]["customerID"] if not top50_df.empty else "N/A"

m1.metric("Total At-Risk Count", f"{at_risk_count:,}")
m2.metric("Monthly Revenue At Risk", f"${monthly_revenue_at_risk:,.2f}")
m3.metric("Average Risk Score", f"{avg_risk_score:.1f} / 100")
m4.metric("Highest-Risk Customer", top_customer)

st.divider()
st.subheader("Visualizations")

tab1, tab2, tab3 = st.tabs([
    "Risk Distribution",
    "Top 50 Watch-List",
    "Risk Factor Contribution",
])

with tab1:
    st.plotly_chart(chart_histogram(scored_df), use_container_width=True)

with tab2:
    st.plotly_chart(chart_top50(top50_df), use_container_width=True)
    display_top50 = top50_df[[
        "rank",
        "customerID",
        "churn_risk_score",
        "risk_band",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "PaymentMethod",
        "Churn",
    ]].copy()
    display_top50["churn_risk_score"] = display_top50["churn_risk_score"].round(2)
    st.dataframe(display_top50, use_container_width=True, hide_index=True)

with tab3:
    st.plotly_chart(chart_contributions(contribution_series), use_container_width=True)

st.divider()
st.subheader("💡 What This Means for Your Business")
very_high_count = (scored_df["risk_band"] == "Very High").sum()
electronic_check_share = ((top50_df["PaymentMethod"] == "Electronic check").mean() * 100) if not top50_df.empty else 0
month_to_month_share = ((top50_df["Contract"] == "Month-to-month").mean() * 100) if not top50_df.empty else 0

st.info(
    f"This score ranks customers by churn risk using contract type, tenure, monthly charges, "
    f"total charges, tech support status, and payment method.\n\n" 
    f"In the current filtered view,"
    f"{very_high_count} customers fall into the Very High risk band. Among the top 50 at-risk customers,"
    f"{month_to_month_share:.1f}% are on month-to-month contracts and {electronic_check_share:.1f}% use "
    f"electronic checks.\n\n"
    f"That pattern suggests the strongest retention focus should be placed on short-term "
    f"customers with weaker service commitment and less stable payment behavior."
)

st.divider()
csv_output = scored_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download scored customer file",
    data=csv_output,
    file_name="telco_churn_scored_results.csv",
    mime="text/csv",
)

st.caption("Built from your Assignment 3 churn notebook and adapted into a deployable Streamlit app.")
