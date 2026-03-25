# ============================================================================
# BUSI3013 Lab 4 Tutorial — Product Velocity Index (PVI) Analyzer
# Maple & Grind Coffee Shop Analytics Tool
# ============================================================================
# This app accepts a CSV of coffee shop transaction data, computes a
# Product Velocity Index (PVI) for every menu item, classifies items
# into performance tiers, and delivers visual results a business owner
# can act on.
#
# Required CSV columns:
#   date, product_category, product_name, quantity, unit_price
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PVI Analyzer | Maple & Grind",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# STUDENT NOTE: List every column the metric formula depends on.
# This list drives the validation check — if any column is missing
# from the uploaded file, the app stops and tells the user exactly
# what is missing rather than crashing with a confusing error.
REQUIRED_COLUMNS = ['date', 'product_category', 'product_name', 'quantity', 'unit_price']

# STUDENT NOTE: Metric weights — defined once here so they are easy to
# adjust and clearly visible without hunting through formula code below.
W_REVENUE    = 0.40
W_FREQUENCY  = 0.40
W_AVG_REV    = 0.20

# STUDENT NOTE: Tier colour palette used consistently across all charts
TIER_COLORS = {
    'Star':      '#2563EB',
    'Performer': '#16A34A',
    'Standard':  '#CA8A04',
    'Review':    '#DC2626',
}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> list:
    """Return a list of required column names that are absent from df."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal cleaning to the uploaded DataFrame.
    STUDENT NOTE: We drop rows only where critical numeric columns are null
    because a null in quantity or unit_price would produce NaN revenue values
    that silently corrupt the metric without raising an error.
    """
    # STUDENT NOTE: Parse date column — coerce errors means bad dates become NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # STUDENT NOTE: Cast numeric columns and drop rows where they are null
    df['quantity']   = pd.to_numeric(df['quantity'],   errors='coerce')
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
    df = df.dropna(subset=['quantity', 'unit_price'])

    # STUDENT NOTE: Compute revenue as a derived column so the app works
    # whether or not the uploaded CSV includes a revenue column
    df['revenue'] = df['quantity'] * df['unit_price']

    return df


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a Series to the 0–100 range using min-max scaling.
    STUDENT NOTE: Returns 50 for all values if the series has no variance
    (e.g., only one product) to avoid division by zero.
    """
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - min_val) / (max_val - min_val) * 100).round(2)


def compute_pvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Product Velocity Index for every product in df.

    Steps:
      1. Aggregate transactions to the product level
      2. Normalize each of the three dimensions to 0–100
      3. Apply weights and compute the composite PVI score
      4. Classify products into performance tiers

    STUDENT NOTE: This function is self-contained — it takes a cleaned
    transaction DataFrame and returns a product-level results DataFrame.
    This design means the same function can be called for the full dataset
    or for any filtered subset (e.g., one store location) without changes.
    """
    # ── Step 1: Aggregate ───────────────────────────────────────────────────
    summary = df.groupby(['product_name', 'product_category']).agg(
        total_revenue       =('revenue',        'sum'),
        transaction_count   =('product_name',   'count'),
        avg_revenue_per_txn =('revenue',        'mean')
    ).round(2).reset_index()

    # ── Step 2: Normalize ───────────────────────────────────────────────────
    summary['revenue_score']   = min_max_normalize(summary['total_revenue'])
    summary['frequency_score'] = min_max_normalize(summary['transaction_count'])
    summary['avg_rev_score']   = min_max_normalize(summary['avg_revenue_per_txn'])

    # ── Step 3: Weighted composite score ───────────────────────────────────
    summary['pvi_score'] = (
        W_REVENUE   * summary['revenue_score']
      + W_FREQUENCY * summary['frequency_score']
      + W_AVG_REV   * summary['avg_rev_score']
    ).round(2)

    # ── Step 4: Tier classification ─────────────────────────────────────────
    # STUDENT NOTE: Percentile-based cutoffs ensure the tiers adapt to any
    # dataset — not just the specific file used during development
    p20 = summary['pvi_score'].quantile(0.20)
    p50 = summary['pvi_score'].quantile(0.50)
    p80 = summary['pvi_score'].quantile(0.80)

    def assign_tier(score):
        if score >= p80:   return 'Star'
        elif score >= p50: return 'Performer'
        elif score >= p20: return 'Standard'
        else:              return 'Review'

    summary['tier'] = summary['pvi_score'].apply(assign_tier)
    summary = summary.sort_values('pvi_score', ascending=False).reset_index(drop=True)
    summary['rank'] = range(1, len(summary) + 1)

    return summary


# ── CHART FUNCTIONS ───────────────────────────────────────────────────────────

def chart_ranking(summary: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart ranking all products by PVI score."""
    plot_df = summary.sort_values('pvi_score', ascending=True)
    fig = px.bar(
        plot_df,
        x='pvi_score',
        y='product_name',
        color='tier',
        color_discrete_map=TIER_COLORS,
        orientation='h',
        text='pvi_score',
        title='Product Velocity Index — All Menu Items Ranked',
        labels={'pvi_score': 'PVI Score (0–100)', 'product_name': 'Product'},
        template='plotly_white',
        hover_data=['product_category', 'total_revenue', 'transaction_count']
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        height=max(400, len(summary) * 30 + 100),
        xaxis_range=[0, 115],
        legend_title_text='Performance Tier',
        title_font_size=14
    )
    return fig


def chart_tier_donut(summary: pd.DataFrame) -> go.Figure:
    """Donut chart showing count and proportion by performance tier."""
    tier_order  = ['Star', 'Performer', 'Standard', 'Review']
    tier_counts = (
        summary['tier']
        .value_counts()
        .reindex(tier_order, fill_value=0)
        .reset_index()
    )
    tier_counts.columns = ['tier', 'count']

    fig = px.pie(
        tier_counts,
        values='count',
        names='tier',
        color='tier',
        color_discrete_map=TIER_COLORS,
        title='Menu Performance Tier Distribution',
        hole=0.45,
        template='plotly_white'
    )
    fig.update_traces(
        textposition='outside',
        textinfo='label+percent+value',
        textfont_size=12
    )
    fig.update_layout(title_font_size=14)
    return fig


def chart_scatter(summary: pd.DataFrame) -> go.Figure:
    """Revenue vs frequency scatter with PVI score as bubble size."""
    median_txn = summary['transaction_count'].median()
    median_rev = summary['total_revenue'].median()

    fig = px.scatter(
        summary,
        x='transaction_count',
        y='total_revenue',
        size='pvi_score',
        color='tier',
        color_discrete_map=TIER_COLORS,
        text='product_name',
        title='Revenue vs. Transaction Frequency — Bubble Size = PVI Score',
        labels={
            'transaction_count': 'Transaction Frequency (# orders)',
            'total_revenue':      'Total Revenue (CAD)',
            'pvi_score':          'PVI Score'
        },
        template='plotly_white',
        size_max=45,
        hover_data=['pvi_score', 'tier']
    )
    # STUDENT NOTE: Median reference lines divide the chart into four
    # interpretable quadrants without needing a written explanation
    fig.add_vline(x=median_txn, line_dash='dash', line_color='grey', opacity=0.5,
                  annotation_text='Median frequency', annotation_position='top right')
    fig.add_hline(y=median_rev, line_dash='dash', line_color='grey', opacity=0.5,
                  annotation_text='Median revenue', annotation_position='top right')
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(height=520, title_font_size=14)
    return fig


def chart_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend line broken down by product category."""
    # STUDENT NOTE: Extract year-month so the x-axis shows periods, not individual dates
    trend_df = df.copy()
    trend_df['month'] = trend_df['date'].dt.to_period('M').astype(str)
    monthly = (
        trend_df.groupby(['month', 'product_category'])['revenue']
        .sum()
        .round(2)
        .reset_index()
    )
    fig = px.line(
        monthly,
        x='month',
        y='revenue',
        color='product_category',
        title='Monthly Revenue Trend by Product Category',
        labels={'month': 'Month', 'revenue': 'Total Revenue (CAD)', 'product_category': 'Category'},
        template='plotly_white',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, title_font_size=14, legend_title_text='Category')
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/coffee-to-go.png", width=60)
    st.title("PVI Analyzer")
    st.caption("Product Velocity Index — Maple & Grind")
    st.divider()

    st.markdown("### How to use")
    st.markdown(
        "1. Upload a transaction CSV  \n"
        "2. Use the filters to focus the analysis  \n"
        "3. Read the interpretation panels  \n"
        "4. Use the results table to plan menu changes"
    )
    st.divider()

    st.markdown("### Required columns")
    for col in REQUIRED_COLUMNS:
        st.markdown(f"- `{col}`")
    st.divider()

    st.markdown("### Score weights")
    st.markdown(
        f"- Revenue contribution: **{int(W_REVENUE*100)}%**  \n"
        f"- Transaction frequency: **{int(W_FREQUENCY*100)}%**  \n"
        f"- Avg revenue per order: **{int(W_AVG_REV*100)}%**"
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("☕ Product Velocity Index Analyzer")
st.markdown(
    "Upload your transaction data to score every menu item on revenue contribution, "
    "transaction frequency, and average order value. Results are classified into four "
    "performance tiers so you can make immediate, evidence-based menu decisions."
)

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your transaction CSV", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file above to begin the analysis.")

    # STUDENT NOTE: Show a sample of what the expected data looks like
    # so users know the correct format before uploading
    st.markdown("#### Expected data format")
    sample = pd.DataFrame({
        'date':             ['2024-01-15', '2024-01-15', '2024-01-16'],
        'product_category': ['Coffee',     'Bakery',     'Tea'],
        'product_name':     ['Latte',      'Croissant',  'Chai Latte'],
        'quantity':         [1,             2,            1],
        'unit_price':       [5.50,          3.75,         5.25],
    })
    st.dataframe(sample, use_container_width=True)
    st.stop()

# ── LOAD AND VALIDATE ─────────────────────────────────────────────────────────
raw_df = pd.read_csv(uploaded_file)

# STUDENT NOTE: Validate columns before doing anything else —
# a clear error message here saves minutes of confusing debugging
missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(
        f"**Missing required columns:** {missing_cols}  \n"
        f"Your file has: {list(raw_df.columns)}  \n"
        f"Please check column names and re-upload."
    )
    st.stop()

df_clean = clean_data(raw_df)

# ── DATA PREVIEW ─────────────────────────────────────────────────────────────
with st.expander("📋 Data Preview (first 10 rows)", expanded=False):
    st.dataframe(df_clean.head(10), use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Transactions", f"{len(df_clean):,}")
    col_b.metric("Products", df_clean['product_name'].nunique())
    col_c.metric("Date Range",
                 f"{df_clean['date'].min().strftime('%b %d')} – "
                 f"{df_clean['date'].max().strftime('%b %d, %Y')}")

# ── INTERACTIVE FILTERS ───────────────────────────────────────────────────────
st.divider()
st.subheader("🔧 Analysis Filters")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    # STUDENT NOTE: Category filter — "All Categories" passes the full dataset;
    # selecting one category runs the metric only on that subset.
    # This is a meaningful filter because it changes which products are included
    # in the aggregation, which changes the normalization bounds, which changes scores.
    all_categories = sorted(df_clean['product_category'].unique())
    category_choice = st.selectbox(
        "Filter by Product Category",
        options=["All Categories"] + all_categories,
        help="Selecting a category scores products within that category only."
    )

with filter_col2:
    # STUDENT NOTE: If the dataset has a store_location column, offer that as a filter too.
    # We check for it rather than requiring it so the app works on simpler datasets.
    if 'store_location' in df_clean.columns:
        all_locations  = sorted(df_clean['store_location'].unique())
        location_choice = st.selectbox(
            "Filter by Store Location",
            options=["All Locations"] + list(all_locations)
        )
    else:
        location_choice = "All Locations"

# STUDENT NOTE: Apply filters to produce the working DataFrame for this analysis run
filtered_df = df_clean.copy()
if category_choice != "All Categories":
    filtered_df = filtered_df[filtered_df['product_category'] == category_choice]
if location_choice != "All Locations" and 'store_location' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['store_location'] == location_choice]

if len(filtered_df) == 0:
    st.warning("No data matches the current filter combination. Please adjust your filters.")
    st.stop()

# ── COMPUTE METRIC ────────────────────────────────────────────────────────────
summary = compute_pvi(filtered_df)

# ── HEADLINE METRICS ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Headline Results")

# STUDENT NOTE: st.metric() is the right widget for KPI-style headline numbers.
# Use delta to show a comparison (here: how many products are in each extreme tier).
m1, m2, m3, m4, m5 = st.columns(5)

top_product    = summary.iloc[0]['product_name']
avg_pvi        = summary['pvi_score'].mean()
star_count     = (summary['tier'] == 'Star').sum()
review_count   = (summary['tier'] == 'Review').sum()
total_revenue  = filtered_df['revenue'].sum()

m1.metric("Top Product",        top_product)
m2.metric("Average PVI Score",  f"{avg_pvi:.1f} / 100")
m3.metric("⭐ Star Items",      star_count,
          help="Top 20% by combined revenue and frequency")
m4.metric("⚠️ Review Items",   review_count,
          help="Bottom 20% — candidates for removal or reformulation")
m5.metric("Total Revenue",      f"${total_revenue:,.0f}")

# ── CHARTS ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "PVI Ranking", "Tier Distribution", "Revenue vs. Frequency", "Revenue Trend"
])

with tab1:
    st.plotly_chart(chart_ranking(summary), use_container_width=True)

with tab2:
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.plotly_chart(chart_tier_donut(summary), use_container_width=True)
    with col_r:
        st.markdown("#### Tier Definitions")
        tier_defs = {
            "⭐ Star":       "Top 20% — highest combined revenue and frequency. Protect these.",
            "✅ Performer":  "50th–80th percentile — solid contributors. Maintain and upsell.",
            "🔄 Standard":   "20th–50th percentile — acceptable but not driving growth. Monitor.",
            "⚠️ Review":    "Bottom 20% — low frequency and revenue. Evaluate for removal.",
        }
        for tier, definition in tier_defs.items():
            st.markdown(f"**{tier}:** {definition}")

with tab3:
    st.plotly_chart(chart_scatter(summary), use_container_width=True)

with tab4:
    if df_clean['date'].notna().sum() > 0:
        st.plotly_chart(chart_revenue_trend(filtered_df), use_container_width=True)
    else:
        st.info("Revenue trend requires a valid date column.")

# ── RESULTS TABLE ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Full Results Table")

display_cols = {
    'rank':                 'Rank',
    'product_name':         'Product',
    'product_category':     'Category',
    'tier':                 'Tier',
    'pvi_score':            'PVI Score',
    'total_revenue':        'Total Revenue',
    'transaction_count':    'Orders',
    'avg_revenue_per_txn':  'Avg Order Value',
}

results_display = summary[list(display_cols.keys())].rename(columns=display_cols)
results_display['Total Revenue']    = results_display['Total Revenue'].map('${:,.2f}'.format)
results_display['Avg Order Value']  = results_display['Avg Order Value'].map('${:.2f}'.format)
results_display['PVI Score']        = results_display['PVI Score'].map('{:.1f}'.format)

st.dataframe(results_display, use_container_width=True, hide_index=True)

# ── INTERPRETATION PANEL ─────────────────────────────────────────────────────
st.divider()
st.subheader("💡 What This Means for Your Business")

star_products   = summary[summary['tier'] == 'Star']['product_name'].tolist()
review_products = summary[summary['tier'] == 'Review']['product_name'].tolist()

# STUDENT NOTE: The interpretation panel translates metric output into
# plain-English business language. This text is written by the analyst —
# it is not generated by AI. It must be specific to the actual results.
st.info(
    f"**Your top-performing items** ({', '.join(star_products)}) drive the highest combination "
    f"of order volume and revenue. These are your core menu — removing or changing them carries "
    f"significant revenue risk. Focus promotions and upsell efforts here first.\n\n"
    f"**Items flagged for review** ({', '.join(review_products) if review_products else 'None'}) "
    f"are ordered infrequently and generate low total revenue relative to other items. "
    f"Before removing any of them, check whether they serve a strategic purpose (e.g., seasonal "
    f"draw, dietary requirement, brand differentiator) that the transaction data does not capture.\n\n"
    f"**The Revenue vs. Frequency chart** is the most useful view for a menu conversation. "
    f"Items in the top-right quadrant (high frequency and high revenue) are unambiguous Stars. "
    f"Items in the bottom-left quadrant are the clearest candidates for discontinuation."
)

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
st.divider()
csv_output = summary.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download PVI Results as CSV",
    data=csv_output,
    file_name="pvi_results.csv",
    mime="text/csv"
)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.caption("BUSI3013 Lab 4 Tutorial — Product Velocity Index Analyzer | Built with Streamlit + Plotly")
