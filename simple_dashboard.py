"""
Simple E-commerce Analytics Dashboard
A working Streamlit dashboard for E-commerce Sales Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

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

def generate_sample_data():
    """Generate sample e-commerce data"""
    np.random.seed(42)
    
    # Generate sample sales data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    n_days = len(dates)
    
    # Generate daily sales data
    sales_data = []
    for i, date in enumerate(dates):
        # Add some seasonality and trends
        base_sales = 1000 + i * 2  # Growing trend
        seasonality = 200 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
        weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
        noise = np.random.normal(0, 100)
        
        daily_revenue = base_sales + seasonality + weekly_pattern + noise
        daily_orders = int(daily_revenue / np.random.uniform(50, 150))
        daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
        
        sales_data.append({
            'date': date,
            'revenue': max(0, daily_revenue),
            'orders': max(1, daily_orders),
            'customers': max(1, daily_customers),
            'avg_order_value': daily_revenue / max(1, daily_orders)
        })
    
    return pd.DataFrame(sales_data)

def generate_product_data():
    """Generate sample product performance data"""
    products = [
        'iPhone 15 Pro', 'Samsung Galaxy S24', 'MacBook Pro', 'Dell XPS 13',
        'AirPods Pro', 'Sony WH-1000XM5', 'iPad Air', 'Surface Pro 9',
        'Nintendo Switch', 'PlayStation 5', 'Xbox Series X', 'Apple Watch',
        'Fitbit Versa', 'GoPro Hero', 'Canon EOS R5', 'Nikon Z9',
        'Tesla Model 3', 'BMW i3', 'Mercedes EQS', 'Audi e-tron'
    ]
    
    categories = ['Electronics', 'Computers', 'Audio', 'Gaming', 'Wearables', 'Cameras', 'Automotive']
    
    product_data = []
    for product in products:
        category = random.choice(categories)
        revenue = np.random.uniform(10000, 500000)
        orders = int(revenue / np.random.uniform(100, 1000))
        margin = np.random.uniform(0.1, 0.4)
        
        product_data.append({
            'product': product,
            'category': category,
            'revenue': revenue,
            'orders': orders,
            'margin': margin,
            'profit': revenue * margin
        })
    
    return pd.DataFrame(product_data)

def generate_customer_data():
    """Generate sample customer segmentation data"""
    segments = ['Champions', 'Loyal Customers', 'New Customers', 'At Risk', 'Cannot Lose Them']
    
    customer_data = []
    for segment in segments:
        if segment == 'Champions':
            count = 150
            avg_spent = 2500
            avg_orders = 25
        elif segment == 'Loyal Customers':
            count = 300
            avg_spent = 1200
            avg_orders = 15
        elif segment == 'New Customers':
            count = 200
            avg_spent = 300
            avg_orders = 3
        elif segment == 'At Risk':
            count = 100
            avg_spent = 800
            avg_orders = 8
        else:  # Cannot Lose Them
            count = 50
            avg_spent = 5000
            avg_orders = 50
        
        customer_data.append({
            'segment': segment,
            'count': count,
            'avg_spent': avg_spent,
            'avg_orders': avg_orders,
            'total_value': count * avg_spent
        })
    
    return pd.DataFrame(customer_data)

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
    
    # Generate sample data
    sales_df = generate_sample_data()
    products_df = generate_product_data()
    customers_df = generate_customer_data()
    
    # Filter data by date range
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    filtered_sales = sales_df[
        (sales_df['date'].dt.date >= start_date) & 
        (sales_df['date'].dt.date <= end_date)
    ]
    
    # If no data in filtered range, show all data
    if len(filtered_sales) == 0:
        st.warning("⚠️ No data found for the selected date range. Showing all available data.")
        filtered_sales = sales_df
    
    # KPI Metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_sales['revenue'].sum()
        st.metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.2f}",
            delta=f"+{np.random.uniform(5, 15):.1f}%"
        )
    
    with col2:
        total_orders = filtered_sales['orders'].sum()
        st.metric(
            label="🛒 Total Orders",
            value=f"{total_orders:,}",
            delta=f"+{np.random.uniform(8, 20):.1f}%"
        )
    
    with col3:
        avg_order_value = filtered_sales['avg_order_value'].mean()
        st.metric(
            label="📊 Avg Order Value",
            value=f"${avg_order_value:.2f}",
            delta=f"+{np.random.uniform(2, 8):.1f}%"
        )
    
    with col4:
        unique_customers = filtered_sales['customers'].sum()
        st.metric(
            label="👥 Unique Customers",
            value=f"{unique_customers:,}",
            delta=f"+{np.random.uniform(3, 12):.1f}%"
        )
    
    # Charts
    st.subheader("📈 Sales Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        fig_revenue = px.line(
            filtered_sales, 
            x='date', 
            y='revenue',
            title='Daily Revenue Trend',
            color_discrete_sequence=['#1f77b4']
        )
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, width='stretch')
    
    with col2:
        # Orders trend
        fig_orders = px.line(
            filtered_sales, 
            x='date', 
            y='orders',
            title='Daily Orders Trend',
            color_discrete_sequence=['#ff7f0e']
        )
        fig_orders.update_layout(height=400)
        st.plotly_chart(fig_orders, width='stretch')
    
    # Product Performance
    st.subheader("🛍️ Product Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products by revenue
        top_products = products_df.nlargest(10, 'revenue')
        fig_products = px.bar(
            top_products,
            x='revenue',
            y='product',
            orientation='h',
            title='Top 10 Products by Revenue',
            color='category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_products.update_layout(height=500)
        st.plotly_chart(fig_products, width='stretch')
    
    with col2:
        # Revenue by category
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
    
    # Customer Analysis
    st.subheader("👥 Customer Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segments
        fig_segments = px.bar(
            customers_df,
            x='segment',
            y='count',
            title='Customer Distribution by Segment',
            color='segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_segments.update_layout(height=400)
        st.plotly_chart(fig_segments, width='stretch')
    
    with col2:
        # Average spending by segment
        fig_spending = px.bar(
            customers_df,
            x='segment',
            y='avg_spent',
            title='Average Spending by Segment',
            color='segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_spending.update_layout(height=400)
        st.plotly_chart(fig_spending, width='stretch')
    
    # Real-time Metrics (Simulated)
    st.subheader("⚡ Real-time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔥 Live Events",
            value=f"{random.randint(100, 1000):,}",
            delta="Real-time"
        )
    
    with col2:
        st.metric(
            label="💰 Live Revenue",
            value=f"${random.uniform(1000, 5000):,.2f}",
            delta="Real-time"
        )
    
    with col3:
        st.metric(
            label="🎯 Live Conversion",
            value=f"{random.uniform(2, 8):.1f}%",
            delta="Real-time"
        )
    
    with col4:
        st.metric(
            label="📊 Live AOV",
            value=f"${random.uniform(80, 200):.2f}",
            delta="Real-time"
        )
    
    # Data Tables
    st.subheader("📋 Detailed Data")
    
    tab1, tab2, tab3 = st.tabs(["Sales Data", "Product Performance", "Customer Segments"])
    
    with tab1:
        st.dataframe(filtered_sales.head(20), width='stretch')
    
    with tab2:
        st.dataframe(products_df.head(20), width='stretch')
    
    with tab3:
        st.dataframe(customers_df, width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**E-commerce Sales Analytics Platform** | "
        "Built with Streamlit and Plotly | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
