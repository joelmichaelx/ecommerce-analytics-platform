"""
Vercel-compatible Streamlit App
E-commerce Sales Analytics Dashboard for Vercel deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os

# Configure page
st.set_page_config(
    page_title="E-commerce Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def get_api_data(endpoint):
    """Get data from Vercel API"""
    try:
        # For local development
        if os.getenv('VERCEL') is None:
            base_url = "http://localhost:8000"
        else:
            # For Vercel deployment
            base_url = "https://ecommerce-analytics-platform-ivory.vercel.app"
        
        response = requests.get(f"{base_url}/api{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def generate_fallback_data():
    """Generate fallback data if API is not available"""
    np.random.seed(42)
    
    # Generate sample sales data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sales_data = []
    
    for i, date in enumerate(dates):
        base_sales = 1000 + i * 2
        seasonality = 200 * np.sin(2 * np.pi * i / 365)
        weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 100)
        
        daily_revenue = max(0, base_sales + seasonality + weekly_pattern + noise)
        daily_orders = int(daily_revenue / np.random.uniform(50, 150))
        daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
        
        sales_data.append({
            'date': date,
            'revenue': daily_revenue,
            'orders': daily_orders,
            'customers': daily_customers,
            'avg_order_value': daily_revenue / max(1, daily_orders)
        })
    
    return pd.DataFrame(sales_data)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 E-commerce Sales Analytics Platform</h1>', 
               unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2024, 1, 1),
        key="start_date"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2024, 12, 31),
        key="end_date"
    )
    
    # Category filter
    st.sidebar.subheader("🏷️ Category Filter")
    categories = ['Electronics', 'Computers', 'Audio', 'Gaming', 'Wearables', 'Cameras', 'Automotive']
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=categories[:3]
    )
    
    # Try to get data from API, fallback to local generation
    sales_summary = get_api_data('/analytics/sales/summary')
    sales_trends = get_api_data('/analytics/sales/trends')
    top_products = get_api_data('/analytics/products/top')
    customer_segments = get_api_data('/analytics/customers/segments')
    realtime_metrics = get_api_data('/realtime/metrics')
    
    # If API is not available, use fallback data
    if sales_summary is None:
        st.warning("⚠️ API not available. Using sample data.")
        sales_df = generate_fallback_data()
        
        # Filter data by date range
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        filtered_sales = sales_df[
            (sales_df['date'].dt.date >= start_date) & 
            (sales_df['date'].dt.date <= end_date)
        ]
        
        if len(filtered_sales) == 0:
            st.warning("⚠️ No data found for the selected date range. Showing all available data.")
            filtered_sales = sales_df
        
        # Calculate summary metrics
        total_revenue = filtered_sales['revenue'].sum()
        total_orders = filtered_sales['orders'].sum()
        unique_customers = filtered_sales['customers'].sum()
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        sales_summary = {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "total_customers": unique_customers,
            "avg_order_value": avg_order_value
        }
        
        # Generate trends data
        trends_data = []
        for _, row in filtered_sales.iterrows():
            trends_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "revenue": row['revenue'],
                "orders": row['orders'],
                "customers": row['customers']
            })
        sales_trends = {"trends": trends_data}
    
    # KPI Metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${sales_summary['total_revenue']:,.2f}",
            delta=f"+{np.random.uniform(5, 15):.1f}%"
        )
    
    with col2:
        st.metric(
            label="🛒 Total Orders",
            value=f"{sales_summary['total_orders']:,}",
            delta=f"+{np.random.uniform(8, 20):.1f}%"
        )
    
    with col3:
        st.metric(
            label="📊 Avg Order Value",
            value=f"${sales_summary['avg_order_value']:.2f}",
            delta=f"+{np.random.uniform(2, 8):.1f}%"
        )
    
    with col4:
        st.metric(
            label="👥 Unique Customers",
            value=f"{sales_summary['total_customers']:,}",
            delta=f"+{np.random.uniform(3, 12):.1f}%"
        )
    
    # Charts
    st.subheader("📈 Sales Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        if sales_trends and 'trends' in sales_trends:
            trends_df = pd.DataFrame(sales_trends['trends'])
            trends_df['date'] = pd.to_datetime(trends_df['date'])
            
            fig_revenue = px.line(
                trends_df, 
                x='date', 
                y='revenue',
                title='Daily Revenue Trend',
                color_discrete_sequence=['#1f77b4']
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, width='stretch')
        else:
            st.info("📊 Revenue trend data not available")
    
    with col2:
        # Orders trend
        if sales_trends and 'trends' in sales_trends:
            trends_df = pd.DataFrame(sales_trends['trends'])
            trends_df['date'] = pd.to_datetime(trends_df['date'])
            
            fig_orders = px.line(
                trends_df, 
                x='date', 
                y='orders',
                title='Daily Orders Trend',
                color_discrete_sequence=['#ff7f0e']
            )
            fig_orders.update_layout(height=400)
            st.plotly_chart(fig_orders, width='stretch')
        else:
            st.info("📊 Orders trend data not available")
    
    # Product Performance
    st.subheader("🛍️ Product Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if top_products and 'top_products' in top_products:
            products_df = pd.DataFrame(top_products['top_products'])
            fig_products = px.bar(
                products_df.head(10),
                x='revenue',
                y='product',
                orientation='h',
                title='Top 10 Products by Revenue',
                color='category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_products.update_layout(height=500)
            st.plotly_chart(fig_products, width='stretch')
        else:
            st.info("🛍️ Product performance data not available")
    
    with col2:
        if top_products and 'top_products' in top_products:
            products_df = pd.DataFrame(top_products['top_products'])
            category_revenue = products_df.groupby('category')['revenue'].sum().reset_index()
            fig_category = px.pie(
                category_revenue,
                values='revenue',
                names='category',
                title='Revenue by Category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_category.update_layout(height=500)
            st.plotly_chart(fig_category, width='stretch')
        else:
            st.info("📊 Category revenue data not available")
    
    # Customer Analysis
    st.subheader("👥 Customer Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if customer_segments and 'customer_segments' in customer_segments:
            segments_df = pd.DataFrame(customer_segments['customer_segments'])
            fig_segments = px.bar(
                segments_df,
                x='segment',
                y='count',
                title='Customer Distribution by Segment',
                color='segment',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_segments.update_layout(height=400)
            st.plotly_chart(fig_segments, width='stretch')
        else:
            st.info("👥 Customer segmentation data not available")
    
    with col2:
        if customer_segments and 'customer_segments' in customer_segments:
            segments_df = pd.DataFrame(customer_segments['customer_segments'])
            fig_spending = px.bar(
                segments_df,
                x='segment',
                y='avg_spent',
                title='Average Spending by Segment',
                color='segment',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_spending.update_layout(height=400)
            st.plotly_chart(fig_spending, width='stretch')
        else:
            st.info("💰 Customer spending data not available")
    
    # Real-time Metrics
    st.subheader("⚡ Real-time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if realtime_metrics:
        with col1:
            st.metric(
                label="🔥 Live Events",
                value=f"{realtime_metrics.get('total_events', 0):,}",
                delta="Real-time"
            )
        
        with col2:
            st.metric(
                label="💰 Live Revenue",
                value=f"${realtime_metrics.get('total_revenue', 0):,.2f}",
                delta="Real-time"
            )
        
        with col3:
            st.metric(
                label="🎯 Live Conversion",
                value=f"{realtime_metrics.get('conversion_rate', 0):.1f}%",
                delta="Real-time"
            )
        
        with col4:
            st.metric(
                label="📊 Live AOV",
                value=f"${realtime_metrics.get('avg_order_value', 0):.2f}",
                delta="Real-time"
            )
    else:
        # Fallback real-time metrics
        with col1:
            st.metric(
                label="🔥 Live Events",
                value=f"{np.random.randint(100, 1000):,}",
                delta="Simulated"
            )
        
        with col2:
            st.metric(
                label="💰 Live Revenue",
                value=f"${np.random.uniform(1000, 5000):,.2f}",
                delta="Simulated"
            )
        
        with col3:
            st.metric(
                label="🎯 Live Conversion",
                value=f"{np.random.uniform(2, 8):.1f}%",
                delta="Simulated"
            )
        
        with col4:
            st.metric(
                label="📊 Live AOV",
                value=f"${np.random.uniform(80, 200):.2f}",
                delta="Simulated"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**E-commerce Sales Analytics Platform** | "
        "Deployed on Vercel | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
