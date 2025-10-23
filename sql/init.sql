-- E-commerce Analytics Database Schema
-- Created for comprehensive sales analytics platform

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_features;

-- Raw data tables (staging area)
CREATE TABLE IF NOT EXISTS raw_data.customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(20),
    registration_date TIMESTAMP,
    customer_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_data.products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    weight DECIMAL(8,2),
    dimensions VARCHAR(50),
    color VARCHAR(50),
    size VARCHAR(20),
    description TEXT,
    sku VARCHAR(100),
    stock_quantity INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_data.orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    order_date TIMESTAMP,
    ship_date TIMESTAMP,
    delivery_date TIMESTAMP,
    order_status VARCHAR(50),
    payment_method VARCHAR(50),
    payment_status VARCHAR(50),
    shipping_address TEXT,
    billing_address TEXT,
    total_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    shipping_cost DECIMAL(10,2),
    discount_amount DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES raw_data.customers(customer_id)
);

CREATE TABLE IF NOT EXISTS raw_data.order_items (
    order_item_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50),
    product_id VARCHAR(50),
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES raw_data.orders(order_id),
    FOREIGN KEY (product_id) REFERENCES raw_data.products(product_id)
);

CREATE TABLE IF NOT EXISTS raw_data.reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50),
    customer_id VARCHAR(50),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    review_date TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    helpful_votes INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES raw_data.products(product_id),
    FOREIGN KEY (customer_id) REFERENCES raw_data.customers(customer_id)
);

-- Processed data tables (data warehouse)
CREATE TABLE IF NOT EXISTS processed_data.dim_customers (
    customer_key SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    customer_segment VARCHAR(50),
    registration_date DATE,
    is_active BOOLEAN,
    effective_date DATE,
    expiry_date DATE,
    current_flag BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processed_data.dim_products (
    product_key SERIAL PRIMARY KEY,
    product_id VARCHAR(50),
    product_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    margin_percentage DECIMAL(5,2),
    is_active BOOLEAN,
    effective_date DATE,
    expiry_date DATE,
    current_flag BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processed_data.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    month_name VARCHAR(20),
    day INTEGER,
    day_name VARCHAR(20),
    week_of_year INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    holiday_name VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS processed_data.fact_sales (
    sales_key SERIAL PRIMARY KEY,
    order_id VARCHAR(50),
    customer_key INTEGER,
    product_key INTEGER,
    date_key INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    cost_amount DECIMAL(10,2),
    profit_amount DECIMAL(10,2),
    margin_percentage DECIMAL(5,2),
    discount_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    shipping_cost DECIMAL(10,2),
    order_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_key) REFERENCES processed_data.dim_customers(customer_key),
    FOREIGN KEY (product_key) REFERENCES processed_data.dim_products(product_key),
    FOREIGN KEY (date_key) REFERENCES processed_data.dim_date(date_key)
);

-- Analytics tables
CREATE TABLE IF NOT EXISTS analytics.daily_sales_summary (
    date DATE,
    total_orders INTEGER,
    total_revenue DECIMAL(12,2),
    total_profit DECIMAL(12,2),
    avg_order_value DECIMAL(10,2),
    total_customers INTEGER,
    new_customers INTEGER,
    repeat_customers INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS analytics.product_performance (
    product_id VARCHAR(50),
    product_name VARCHAR(255),
    category VARCHAR(100),
    total_sales DECIMAL(12,2),
    total_quantity INTEGER,
    total_orders INTEGER,
    avg_rating DECIMAL(3,2),
    total_reviews INTEGER,
    profit_margin DECIMAL(5,2),
    last_sale_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id)
);

CREATE TABLE IF NOT EXISTS analytics.customer_segments (
    customer_id VARCHAR(50),
    segment VARCHAR(50),
    total_spent DECIMAL(12,2),
    total_orders INTEGER,
    avg_order_value DECIMAL(10,2),
    last_order_date DATE,
    days_since_last_order INTEGER,
    lifetime_value DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id)
);

-- ML Features tables
CREATE TABLE IF NOT EXISTS ml_features.customer_features (
    customer_id VARCHAR(50),
    feature_name VARCHAR(100),
    feature_value DECIMAL(15,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, feature_name)
);

CREATE TABLE IF NOT EXISTS ml_features.product_features (
    product_id VARCHAR(50),
    feature_name VARCHAR(100),
    feature_value DECIMAL(15,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id, feature_name)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON raw_data.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON raw_data.orders(order_date);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON raw_data.order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON raw_data.order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_date_key ON processed_data.fact_sales(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_sales_customer_key ON processed_data.fact_sales(customer_key);
CREATE INDEX IF NOT EXISTS idx_fact_sales_product_key ON processed_data.fact_sales(product_key);

-- Create views for common analytics queries
CREATE OR REPLACE VIEW analytics.monthly_sales_trend AS
SELECT 
    d.year,
    d.month,
    d.month_name,
    COUNT(DISTINCT fs.order_id) as total_orders,
    SUM(fs.total_amount) as total_revenue,
    SUM(fs.profit_amount) as total_profit,
    AVG(fs.total_amount) as avg_order_value
FROM processed_data.fact_sales fs
JOIN processed_data.dim_date d ON fs.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

CREATE OR REPLACE VIEW analytics.top_products AS
SELECT 
    p.product_name,
    p.category,
    p.brand,
    SUM(fs.total_amount) as total_revenue,
    SUM(fs.quantity) as total_quantity,
    COUNT(DISTINCT fs.order_id) as total_orders,
    AVG(fs.margin_percentage) as avg_margin
FROM processed_data.fact_sales fs
JOIN processed_data.dim_products p ON fs.product_key = p.product_key
WHERE p.current_flag = TRUE
GROUP BY p.product_id, p.product_name, p.category, p.brand
ORDER BY total_revenue DESC;

-- Insert sample date dimension data
INSERT INTO processed_data.dim_date (date_key, full_date, year, quarter, month, month_name, day, day_name, week_of_year, is_weekend, is_holiday, holiday_name)
SELECT 
    EXTRACT(EPOCH FROM date_series)::INTEGER as date_key,
    date_series as full_date,
    EXTRACT(YEAR FROM date_series) as year,
    EXTRACT(QUARTER FROM date_series) as quarter,
    EXTRACT(MONTH FROM date_series) as month,
    TO_CHAR(date_series, 'Month') as month_name,
    EXTRACT(DAY FROM date_series) as day,
    TO_CHAR(date_series, 'Day') as day_name,
    EXTRACT(WEEK FROM date_series) as week_of_year,
    EXTRACT(DOW FROM date_series) IN (0, 6) as is_weekend,
    FALSE as is_holiday,
    NULL as holiday_name
FROM generate_series(
    '2020-01-01'::date,
    '2030-12-31'::date,
    '1 day'::interval
) as date_series
ON CONFLICT (date_key) DO NOTHING;
