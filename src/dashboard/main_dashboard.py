"""
E-commerce Sales Analytics Dashboard
Interactive Streamlit dashboard for comprehensive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import requests
from sqlalchemy import create_engine
import redis
import logging

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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class EcommerceDashboard:
    """Main dashboard class for E-commerce Analytics"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_connections()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_connections(self):
        """Setup database and cache connections"""
        try:
            # Database connection
            self.db_engine = create_engine(
                "postgresql://admin:password@localhost:5432/ecommerce_analytics"
            )
            
            # Redis connection
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            self.logger.info("Connections established successfully")
            
        except Exception as e:
            self.logger.error(f"Connection setup failed: {str(e)}")
            st.error("Failed to connect to database. Please check your connections.")
    
    def load_data(self, query: str) -> pd.DataFrame:
        """Load data from database"""
        try:
            return pd.read_sql(query, self.db_engine)
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame()
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from Redis"""
        try:
            metrics = {
                'total_events': int(self.redis_client.get('total_events') or 0),
                'total_revenue': float(self.redis_client.get('total_revenue') or 0),
                'conversion_rate': 0,
                'avg_order_value': 0
            }
            
            # Calculate conversion rate
            total_views = int(self.redis_client.get('events:view') or 0)
            total_purchases = int(self.redis_client.get('events:purchase') or 0)
            if total_views > 0:
                metrics['conversion_rate'] = (total_purchases / total_views) * 100
            
            # Calculate average order value
            if total_purchases > 0:
                metrics['avg_order_value'] = metrics['total_revenue'] / total_purchases
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time metrics: {str(e)}")
            return {}
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 E-commerce Sales Analytics Platform</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with filters and controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                key="start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="end_date"
            )
        
        # Category filter
        st.sidebar.subheader("🏷️ Category Filter")
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'],
            default=['Electronics', 'Clothing', 'Home & Garden']
        )
        
        # Customer segment filter
        st.sidebar.subheader("👥 Customer Segments")
        segments = st.sidebar.multiselect(
            "Select Customer Segments",
            options=['Champions', 'Loyal Customers', 'New Customers', 'At Risk', 'Cannot Lose Them'],
            default=['Champions', 'Loyal Customers']
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", key="refresh_button"):
            st.rerun()
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'categories': categories,
            'segments': segments
        }
    
    def render_kpi_metrics(self, filters: Dict[str, Any]):
        """Render KPI metrics cards"""
        st.subheader("📈 Key Performance Indicators")
        
        # Get real-time metrics
        realtime_metrics = self.get_realtime_metrics()
        
        # Get historical metrics
        query = f"""
        SELECT 
            COUNT(DISTINCT order_id) as total_orders,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_order_value,
            COUNT(DISTINCT customer_key) as unique_customers
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN '{filters['start_date']}' AND '{filters['end_date']}'
        """
        
        historical_data = self.load_data(query)
        
        if not historical_data.empty:
            total_orders = historical_data['total_orders'].iloc[0]
            total_revenue = historical_data['total_revenue'].iloc[0]
            avg_order_value = historical_data['avg_order_value'].iloc[0]
            unique_customers = historical_data['unique_customers'].iloc[0]
        else:
            total_orders = 0
            total_revenue = 0
            avg_order_value = 0
            unique_customers = 0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="🛒 Total Orders",
                value=f"{total_orders:,}",
                delta=f"+{realtime_metrics.get('total_events', 0)} (Real-time)"
            )
        
        with col2:
            st.metric(
                label="💰 Total Revenue",
                value=f"${total_revenue:,.2f}",
                delta=f"+${realtime_metrics.get('total_revenue', 0):.2f} (Real-time)"
            )
        
        with col3:
            st.metric(
                label="📊 Avg Order Value",
                value=f"${avg_order_value:.2f}",
                delta=f"{realtime_metrics.get('avg_order_value', 0):.2f} (Real-time)"
            )
        
        with col4:
            st.metric(
                label="👥 Unique Customers",
                value=f"{unique_customers:,}",
                delta="Active"
            )
        
        with col5:
            st.metric(
                label="🎯 Conversion Rate",
                value=f"{realtime_metrics.get('conversion_rate', 0):.1f}%",
                delta="Real-time"
            )
    
    def render_sales_trends(self, filters: Dict[str, Any]):
        """Render sales trends charts"""
        st.subheader("📈 Sales Trends Analysis")
        
        # Daily sales trend
        daily_trend_query = f"""
        SELECT 
            dd.full_date as date,
            SUM(fs.total_amount) as revenue,
            COUNT(DISTINCT fs.order_id) as orders,
            COUNT(DISTINCT fs.customer_key) as customers
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN '{filters['start_date']}' AND '{filters['end_date']}'
        GROUP BY dd.full_date
        ORDER BY dd.full_date
        """
        
        daily_trend_data = self.load_data(daily_trend_query)
        
        if not daily_trend_data.empty:
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Revenue Trend', 'Daily Orders Trend', 
                              'Daily Customers Trend', 'Revenue vs Orders'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            # Daily revenue
            fig.add_trace(
                go.Scatter(x=daily_trend_data['date'], y=daily_trend_data['revenue'],
                          mode='lines+markers', name='Revenue', line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            # Daily orders
            fig.add_trace(
                go.Scatter(x=daily_trend_data['date'], y=daily_trend_data['orders'],
                          mode='lines+markers', name='Orders', line=dict(color='#ff7f0e')),
                row=1, col=2
            )
            
            # Daily customers
            fig.add_trace(
                go.Scatter(x=daily_trend_data['date'], y=daily_trend_data['customers'],
                          mode='lines+markers', name='Customers', line=dict(color='#2ca02c')),
                row=2, col=1
            )
            
            # Revenue vs Orders scatter
            fig.add_trace(
                go.Scatter(x=daily_trend_data['orders'], y=daily_trend_data['revenue'],
                          mode='markers', name='Revenue vs Orders', 
                          marker=dict(color='#d62728', size=8)),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_product_analysis(self, filters: Dict[str, Any]):
        """Render product performance analysis"""
        st.subheader("🛍️ Product Performance Analysis")
        
        # Top products by revenue
        top_products_query = f"""
        SELECT 
            dp.product_name,
            dp.category,
            SUM(fs.total_amount) as total_revenue,
            SUM(fs.quantity) as total_quantity,
            COUNT(DISTINCT fs.order_id) as total_orders,
            AVG(fs.margin_percentage) as avg_margin
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_products dp ON fs.product_key = dp.product_key
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN '{filters['start_date']}' AND '{filters['end_date']}'
        AND dp.current_flag = TRUE
        GROUP BY dp.product_id, dp.product_name, dp.category
        ORDER BY total_revenue DESC
        LIMIT 20
        """
        
        top_products_data = self.load_data(top_products_query)
        
        if not top_products_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top products bar chart
                fig = px.bar(
                    top_products_data.head(10),
                    x='total_revenue',
                    y='product_name',
                    orientation='h',
                    title='Top 10 Products by Revenue',
                    color='category',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Product categories pie chart
                category_revenue = top_products_data.groupby('category')['total_revenue'].sum().reset_index()
                fig = px.pie(
                    category_revenue,
                    values='total_revenue',
                    names='category',
                    title='Revenue by Category',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_customer_analysis(self, filters: Dict[str, Any]):
        """Render customer analysis"""
        st.subheader("👥 Customer Analysis")
        
        # Customer segments
        customer_segments_query = f"""
        SELECT 
            cs.customer_segment,
            COUNT(*) as customer_count,
            AVG(cs.total_spent) as avg_spent,
            AVG(cs.total_orders) as avg_orders,
            AVG(cs.avg_order_value) as avg_order_value
        FROM analytics.customer_segments cs
        WHERE cs.customer_segment IN ({','.join([f"'{s}'" for s in filters['segments']])})
        GROUP BY cs.customer_segment
        ORDER BY customer_count DESC
        """
        
        customer_segments_data = self.load_data(customer_segments_query)
        
        if not customer_segments_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Customer segments bar chart
                fig = px.bar(
                    customer_segments_data,
                    x='customer_segment',
                    y='customer_count',
                    title='Customer Distribution by Segment',
                    color='customer_segment',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average spending by segment
                fig = px.bar(
                    customer_segments_data,
                    x='customer_segment',
                    y='avg_spent',
                    title='Average Spending by Segment',
                    color='customer_segment',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_geographic_analysis(self, filters: Dict[str, Any]):
        """Render geographic analysis"""
        st.subheader("🌍 Geographic Analysis")
        
        # Geographic revenue distribution
        geo_query = f"""
        SELECT 
            dc.state,
            dc.country,
            SUM(fs.total_amount) as total_revenue,
            COUNT(DISTINCT fs.customer_key) as unique_customers,
            COUNT(DISTINCT fs.order_id) as total_orders
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_customers dc ON fs.customer_key = dc.customer_key
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN '{filters['start_date']}' AND '{filters['end_date']}'
        AND dc.current_flag = TRUE
        GROUP BY dc.state, dc.country
        ORDER BY total_revenue DESC
        """
        
        geo_data = self.load_data(geo_query)
        
        if not geo_data.empty:
            # Geographic revenue map
            fig = px.choropleth(
                geo_data,
                locations='state',
                locationmode='USA-states',
                color='total_revenue',
                title='Revenue by State',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_ml_insights(self, filters: Dict[str, Any]):
        """Render ML insights and predictions"""
        st.subheader("🤖 Machine Learning Insights")
        
        # Sales forecasting
        st.write("#### 📊 Sales Forecasting")
        
        # Load forecasting data
        forecast_query = f"""
        SELECT 
            dd.full_date as date,
            SUM(fs.total_amount) as revenue,
            COUNT(DISTINCT fs.order_id) as orders
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN '{filters['start_date']}' AND '{filters['end_date']}'
        GROUP BY dd.full_date
        ORDER BY dd.full_date
        """
        
        forecast_data = self.load_data(forecast_query)
        
        if not forecast_data.empty:
            # Simple trend analysis
            forecast_data['revenue_ma7'] = forecast_data['revenue'].rolling(window=7).mean()
            forecast_data['revenue_ma30'] = forecast_data['revenue'].rolling(window=30).mean()
            
            # Create forecast chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['revenue'],
                mode='lines',
                name='Actual Revenue',
                line=dict(color='#1f77b4')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['revenue_ma7'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='#ff7f0e', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['revenue_ma30'],
                mode='lines',
                name='30-Day Moving Average',
                line=dict(color='#2ca02c', dash='dot')
            ))
            
            fig.update_layout(
                title='Revenue Trend with Moving Averages',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer churn prediction
        st.write("#### ⚠️ Customer Churn Analysis")
        
        churn_query = """
        SELECT 
            customer_id,
            churn_probability,
            risk_category,
            days_since_last_order,
            total_spent
        FROM analytics.customer_churn_prediction
        ORDER BY churn_probability DESC
        LIMIT 20
        """
        
        churn_data = self.load_data(churn_query)
        
        if not churn_data.empty:
            # Churn risk distribution
            fig = px.bar(
                churn_data,
                x='customer_id',
                y='churn_probability',
                color='risk_category',
                title='Top 20 Customers by Churn Risk',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_realtime_dashboard(self):
        """Render real-time dashboard"""
        st.subheader("⚡ Real-time Analytics")
        
        # Get real-time data from Redis
        realtime_metrics = self.get_realtime_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        # Auto-refresh every 30 seconds
        if st.button("🔄 Auto-refresh (30s)", key="auto_refresh"):
            st.rerun()
    
    def run_dashboard(self):
        """Main dashboard runner"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get filters
            filters = self.render_sidebar()
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "📈 Sales Trends", "🛍️ Products", 
                "👥 Customers", "🌍 Geography", "🤖 ML Insights"
            ])
            
            with tab1:
                self.render_kpi_metrics(filters)
                self.render_realtime_dashboard()
            
            with tab2:
                self.render_sales_trends(filters)
            
            with tab3:
                self.render_product_analysis(filters)
            
            with tab4:
                self.render_customer_analysis(filters)
            
            with tab5:
                self.render_geographic_analysis(filters)
            
            with tab6:
                self.render_ml_insights(filters)
            
            # Footer
            st.markdown("---")
            st.markdown(
                "**E-commerce Sales Analytics Platform** | "
                "Built with Streamlit, Plotly, and Advanced ML | "
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except Exception as e:
            self.logger.error(f"Dashboard rendering failed: {str(e)}")
            st.error("An error occurred while rendering the dashboard. Please check the logs.")

# Main execution
if __name__ == "__main__":
    dashboard = EcommerceDashboard()
    dashboard.run_dashboard()
