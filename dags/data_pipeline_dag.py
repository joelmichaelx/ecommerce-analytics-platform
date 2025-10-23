"""
E-commerce Sales Analytics Data Pipeline
Comprehensive data processing pipeline using Apache Airflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging

# Default arguments
default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
dag = DAG(
    'ecommerce_sales_analytics_pipeline',
    default_args=default_args,
    description='Comprehensive E-commerce Sales Analytics Pipeline',
    schedule_interval='@daily',
    max_active_runs=1,
    tags=['ecommerce', 'analytics', 'sales', 'ml']
)

def extract_raw_data(**context):
    """Extract data from various sources"""
    logging.info("Starting data extraction...")
    
    # Simulate data extraction from multiple sources
    # In production, this would connect to actual data sources
    
    # Generate sample customer data
    np.random.seed(42)
    n_customers = 1000
    
    customers_data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(1, n_customers + 1)],
        'first_name': np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa'], n_customers),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], n_customers),
        'email': [f'customer{i}@example.com' for i in range(1, n_customers + 1)],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], n_customers),
        'country': 'USA',
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_customers, p=[0.2, 0.6, 0.2]),
        'registration_date': pd.date_range('2020-01-01', '2024-01-01', periods=n_customers)
    }
    
    # Generate sample product data
    n_products = 500
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    
    products_data = {
        'product_id': [f'PROD_{i:06d}' for i in range(1, n_products + 1)],
        'product_name': [f'Product {i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3'], n_products),
        'brand': np.random.choice(brands, n_products),
        'price': np.random.uniform(10, 1000, n_products).round(2),
        'cost': np.random.uniform(5, 500, n_products).round(2),
        'stock_quantity': np.random.randint(0, 1000, n_products),
        'is_active': np.random.choice([True, False], n_products, p=[0.9, 0.1])
    }
    
    # Generate sample orders data
    n_orders = 5000
    order_dates = pd.date_range('2023-01-01', '2024-01-01', periods=n_orders)
    
    orders_data = {
        'order_id': [f'ORDER_{i:08d}' for i in range(1, n_orders + 1)],
        'customer_id': np.random.choice(customers_data['customer_id'], n_orders),
        'order_date': order_dates,
        'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled'], n_orders, p=[0.8, 0.15, 0.05]),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], n_orders),
        'total_amount': np.random.uniform(20, 2000, n_orders).round(2),
        'tax_amount': np.random.uniform(2, 200, n_orders).round(2),
        'shipping_cost': np.random.uniform(0, 50, n_orders).round(2),
        'discount_amount': np.random.uniform(0, 100, n_orders).round(2)
    }
    
    # Generate sample order items
    order_items_data = []
    for order in orders_data['order_id']:
        n_items = np.random.randint(1, 6)
        for _ in range(n_items):
            product_id = np.random.choice(products_data['product_id'])
            quantity = np.random.randint(1, 5)
            unit_price = np.random.uniform(10, 500)
            
            order_items_data.append({
                'order_item_id': f'ITEM_{len(order_items_data) + 1:08d}',
                'order_id': order,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_price': quantity * unit_price,
                'discount_percentage': np.random.uniform(0, 20)
            })
    
    # Store data in context for next tasks
    context['task_instance'].xcom_push(key='customers_data', value=customers_data)
    context['task_instance'].xcom_push(key='products_data', value=products_data)
    context['task_instance'].xcom_push(key='orders_data', value=orders_data)
    context['task_instance'].xcom_push(key='order_items_data', value=order_items_data)
    
    logging.info(f"Extracted {n_customers} customers, {n_products} products, {n_orders} orders")
    return "Data extraction completed successfully"

def load_raw_data(**context):
    """Load extracted data into raw data tables"""
    logging.info("Loading data into raw tables...")
    
    # Get data from previous task
    customers_data = context['task_instance'].xcom_pull(key='customers_data')
    products_data = context['task_instance'].xcom_pull(key='products_data')
    orders_data = context['task_instance'].xcom_pull(key='orders_data')
    order_items_data = context['task_instance'].xcom_pull(key='order_items_data')
    
    # Create database connection
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = create_engine(postgres_hook.get_uri())
    
    # Load customers data
    customers_df = pd.DataFrame(customers_data)
    customers_df.to_sql('customers', engine, schema='raw_data', if_exists='replace', index=False)
    
    # Load products data
    products_df = pd.DataFrame(products_data)
    products_df.to_sql('products', engine, schema='raw_data', if_exists='replace', index=False)
    
    # Load orders data
    orders_df = pd.DataFrame(orders_data)
    orders_df.to_sql('orders', engine, schema='raw_data', if_exists='replace', index=False)
    
    # Load order items data
    order_items_df = pd.DataFrame(order_items_data)
    order_items_df.to_sql('order_items', engine, schema='raw_data', if_exists='replace', index=False)
    
    logging.info("Data loaded successfully into raw tables")
    return "Raw data loading completed"

def transform_and_load_dimensions(**context):
    """Transform and load dimension tables"""
    logging.info("Transforming and loading dimension tables...")
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = create_engine(postgres_hook.get_uri())
    
    # Transform customers dimension
    customers_query = """
    INSERT INTO processed_data.dim_customers 
    (customer_id, first_name, last_name, email, city, state, country, 
     customer_segment, registration_date, is_active, effective_date, current_flag)
    SELECT 
        customer_id,
        first_name,
        last_name,
        email,
        city,
        state,
        country,
        customer_segment,
        registration_date::date,
        TRUE as is_active,
        registration_date::date as effective_date,
        TRUE as current_flag
    FROM raw_data.customers
    ON CONFLICT (customer_id) DO NOTHING;
    """
    
    # Transform products dimension
    products_query = """
    INSERT INTO processed_data.dim_products 
    (product_id, product_name, category, subcategory, brand, price, cost, 
     margin_percentage, is_active, effective_date, current_flag)
    SELECT 
        product_id,
        product_name,
        category,
        subcategory,
        brand,
        price,
        cost,
        ROUND(((price - cost) / price * 100), 2) as margin_percentage,
        is_active,
        created_at::date as effective_date,
        TRUE as current_flag
    FROM raw_data.products
    ON CONFLICT (product_id) DO NOTHING;
    """
    
    # Execute transformations
    with engine.connect() as conn:
        conn.execute(customers_query)
        conn.execute(products_query)
        conn.commit()
    
    logging.info("Dimension tables transformed and loaded")
    return "Dimension transformation completed"

def load_fact_sales(**context):
    """Load fact sales table"""
    logging.info("Loading fact sales table...")
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = create_engine(postgres_hook.get_uri())
    
    fact_sales_query = """
    INSERT INTO processed_data.fact_sales 
    (order_id, customer_key, product_key, date_key, quantity, unit_price, 
     total_amount, cost_amount, profit_amount, margin_percentage, 
     discount_amount, tax_amount, shipping_cost, order_date)
    SELECT 
        oi.order_id,
        dc.customer_key,
        dp.product_key,
        dd.date_key,
        oi.quantity,
        oi.unit_price,
        oi.total_price as total_amount,
        (oi.total_price * dp.cost / dp.price) as cost_amount,
        (oi.total_price - (oi.total_price * dp.cost / dp.price)) as profit_amount,
        dp.margin_percentage,
        (oi.total_price * oi.discount_percentage / 100) as discount_amount,
        o.tax_amount,
        o.shipping_cost,
        o.order_date
    FROM raw_data.order_items oi
    JOIN raw_data.orders o ON oi.order_id = o.order_id
    JOIN processed_data.dim_customers dc ON o.customer_id = dc.customer_id
    JOIN processed_data.dim_products dp ON oi.product_id = dp.product_id
    JOIN processed_data.dim_date dd ON o.order_date::date = dd.full_date
    WHERE dc.current_flag = TRUE AND dp.current_flag = TRUE
    ON CONFLICT DO NOTHING;
    """
    
    with engine.connect() as conn:
        conn.execute(fact_sales_query)
        conn.commit()
    
    logging.info("Fact sales table loaded")
    return "Fact sales loading completed"

def create_analytics_summaries(**context):
    """Create analytics summary tables"""
    logging.info("Creating analytics summaries...")
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = create_engine(postgres_hook.get_uri())
    
    # Daily sales summary
    daily_summary_query = """
    INSERT INTO analytics.daily_sales_summary 
    (date, total_orders, total_revenue, total_profit, avg_order_value, 
     total_customers, new_customers, repeat_customers)
    SELECT 
        dd.full_date as date,
        COUNT(DISTINCT fs.order_id) as total_orders,
        SUM(fs.total_amount) as total_revenue,
        SUM(fs.profit_amount) as total_profit,
        AVG(fs.total_amount) as avg_order_value,
        COUNT(DISTINCT fs.customer_key) as total_customers,
        COUNT(DISTINCT CASE WHEN dc.registration_date::date = dd.full_date THEN fs.customer_key END) as new_customers,
        COUNT(DISTINCT CASE WHEN dc.registration_date::date < dd.full_date THEN fs.customer_key END) as repeat_customers
    FROM processed_data.fact_sales fs
    JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
    JOIN processed_data.dim_customers dc ON fs.customer_key = dc.customer_key
    GROUP BY dd.full_date
    ON CONFLICT (date) DO UPDATE SET
        total_orders = EXCLUDED.total_orders,
        total_revenue = EXCLUDED.total_revenue,
        total_profit = EXCLUDED.total_profit,
        avg_order_value = EXCLUDED.avg_order_value,
        total_customers = EXCLUDED.total_customers,
        new_customers = EXCLUDED.new_customers,
        repeat_customers = EXCLUDED.repeat_customers;
    """
    
    # Product performance
    product_performance_query = """
    INSERT INTO analytics.product_performance 
    (product_id, product_name, category, total_sales, total_quantity, 
     total_orders, profit_margin, last_sale_date)
    SELECT 
        dp.product_id,
        dp.product_name,
        dp.category,
        SUM(fs.total_amount) as total_sales,
        SUM(fs.quantity) as total_quantity,
        COUNT(DISTINCT fs.order_id) as total_orders,
        AVG(fs.margin_percentage) as profit_margin,
        MAX(dd.full_date) as last_sale_date
    FROM processed_data.fact_sales fs
    JOIN processed_data.dim_products dp ON fs.product_key = dp.product_key
    JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
    WHERE dp.current_flag = TRUE
    GROUP BY dp.product_id, dp.product_name, dp.category
    ON CONFLICT (product_id) DO UPDATE SET
        total_sales = EXCLUDED.total_sales,
        total_quantity = EXCLUDED.total_quantity,
        total_orders = EXCLUDED.total_orders,
        profit_margin = EXCLUDED.profit_margin,
        last_sale_date = EXCLUDED.last_sale_date;
    """
    
    with engine.connect() as conn:
        conn.execute(daily_summary_query)
        conn.execute(product_performance_query)
        conn.commit()
    
    logging.info("Analytics summaries created")
    return "Analytics summaries completed"

def generate_ml_features(**context):
    """Generate ML features for machine learning models"""
    logging.info("Generating ML features...")
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = create_engine(postgres_hook.get_uri())
    
    # Customer features
    customer_features_query = """
    INSERT INTO ml_features.customer_features (customer_id, feature_name, feature_value)
    SELECT 
        dc.customer_id,
        'total_spent' as feature_name,
        SUM(fs.total_amount) as feature_value
    FROM processed_data.dim_customers dc
    JOIN processed_data.fact_sales fs ON dc.customer_key = fs.customer_key
    WHERE dc.current_flag = TRUE
    GROUP BY dc.customer_id
    
    UNION ALL
    
    SELECT 
        dc.customer_id,
        'total_orders' as feature_name,
        COUNT(DISTINCT fs.order_id) as feature_value
    FROM processed_data.dim_customers dc
    JOIN processed_data.fact_sales fs ON dc.customer_key = fs.customer_key
    WHERE dc.current_flag = TRUE
    GROUP BY dc.customer_id
    
    UNION ALL
    
    SELECT 
        dc.customer_id,
        'avg_order_value' as feature_name,
        AVG(fs.total_amount) as feature_value
    FROM processed_data.dim_customers dc
    JOIN processed_data.fact_sales fs ON dc.customer_key = fs.customer_key
    WHERE dc.current_flag = TRUE
    GROUP BY dc.customer_id
    ON CONFLICT (customer_id, feature_name) DO UPDATE SET
        feature_value = EXCLUDED.feature_value;
    """
    
    # Product features
    product_features_query = """
    INSERT INTO ml_features.product_features (product_id, feature_name, feature_value)
    SELECT 
        dp.product_id,
        'total_sales' as feature_name,
        SUM(fs.total_amount) as feature_value
    FROM processed_data.dim_products dp
    JOIN processed_data.fact_sales fs ON dp.product_key = fs.product_key
    WHERE dp.current_flag = TRUE
    GROUP BY dp.product_id
    
    UNION ALL
    
    SELECT 
        dp.product_id,
        'total_quantity_sold' as feature_name,
        SUM(fs.quantity) as feature_value
    FROM processed_data.dim_products dp
    JOIN processed_data.fact_sales fs ON dp.product_key = fs.product_key
    WHERE dp.current_flag = TRUE
    GROUP BY dp.product_id
    ON CONFLICT (product_id, feature_name) DO UPDATE SET
        feature_value = EXCLUDED.feature_value;
    """
    
    with engine.connect() as conn:
        conn.execute(customer_features_query)
        conn.execute(product_features_query)
        conn.commit()
    
    logging.info("ML features generated")
    return "ML features generation completed"

# Task definitions
extract_task = PythonOperator(
    task_id='extract_raw_data',
    python_callable=extract_raw_data,
    dag=dag
)

load_raw_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    dag=dag
)

transform_dimensions_task = PythonOperator(
    task_id='transform_dimensions',
    python_callable=transform_and_load_dimensions,
    dag=dag
)

load_fact_task = PythonOperator(
    task_id='load_fact_sales',
    python_callable=load_fact_sales,
    dag=dag
)

analytics_summaries_task = PythonOperator(
    task_id='create_analytics_summaries',
    python_callable=create_analytics_summaries,
    dag=dag
)

ml_features_task = PythonOperator(
    task_id='generate_ml_features',
    python_callable=generate_ml_features,
    dag=dag
)

# Data quality checks
data_quality_check = PostgresOperator(
    task_id='data_quality_check',
    postgres_conn_id='postgres_default',
    sql="""
    -- Check for data quality issues
    SELECT 
        'Data Quality Check' as check_type,
        COUNT(*) as total_records,
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(DISTINCT product_id) as unique_products,
        COUNT(DISTINCT order_id) as unique_orders
    FROM raw_data.orders;
    """,
    dag=dag
)

# Task dependencies
extract_task >> load_raw_task >> transform_dimensions_task >> load_fact_task >> analytics_summaries_task >> ml_features_task
load_raw_task >> data_quality_check
