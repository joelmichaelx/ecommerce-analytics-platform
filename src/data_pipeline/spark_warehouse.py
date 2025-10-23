"""
Apache Spark Data Warehouse
Advanced data processing and analytics using Spark SQL and DataFrames
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

class SparkDataWarehouse:
    """Spark-based data warehouse for large-scale analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.spark = self._create_spark_session()
        self._setup_warehouse()
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized Spark session for data warehouse operations"""
        try:
            spark = SparkSession.builder \
                .appName("EcommerceDataWarehouse") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.warehouse.dir", self.config.get('warehouse_dir', '/tmp/spark-warehouse')) \
                .config("spark.sql.catalogImplementation", "hive") \
                .enableHiveSupport() \
                .getOrCreate()
            
            # Set log level
            spark.sparkContext.setLogLevel("WARN")
            
            self.logger.info("Spark session created successfully")
            return spark
            
        except Exception as e:
            self.logger.error(f"Failed to create Spark session: {str(e)}")
            raise
    
    def _setup_warehouse(self):
        """Setup data warehouse schemas and tables"""
        try:
            # Create database if not exists
            self.spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_warehouse")
            self.spark.sql("USE ecommerce_warehouse")
            
            # Create schemas
            schemas = ['raw_data', 'processed_data', 'analytics', 'ml_features']
            for schema in schemas:
                self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            
            self.logger.info("Data warehouse schemas created")
            
        except Exception as e:
            self.logger.error(f"Failed to setup warehouse: {str(e)}")
            raise
    
    def load_data_from_postgres(self, table_name: str, schema: str = 'raw_data') -> None:
        """Load data from PostgreSQL to Spark"""
        try:
            # PostgreSQL connection properties
            pg_properties = {
                "driver": "org.postgresql.Driver",
                "url": f"jdbc:postgresql://{self.config['postgres_host']}:{self.config['postgres_port']}/{self.config['postgres_db']}",
                "user": self.config['postgres_user'],
                "password": self.config['postgres_password']
            }
            
            # Load data from PostgreSQL
            df = self.spark.read \
                .format("jdbc") \
                .option("url", pg_properties["url"]) \
                .option("dbtable", f"{schema}.{table_name}") \
                .option("user", pg_properties["user"]) \
                .option("password", pg_properties["password"]) \
                .option("driver", pg_properties["driver"]) \
                .load()
            
            # Write to Spark warehouse
            df.write \
                .mode("overwrite") \
                .option("path", f"/tmp/spark-warehouse/ecommerce_warehouse.{schema}.{table_name}") \
                .saveAsTable(f"{schema}.{table_name}")
            
            self.logger.info(f"Successfully loaded {table_name} from PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"Failed to load {table_name} from PostgreSQL: {str(e)}")
            raise
    
    def create_dimension_tables(self):
        """Create and populate dimension tables"""
        try:
            # Load raw data
            customers_df = self.spark.table("raw_data.customers")
            products_df = self.spark.table("raw_data.products")
            
            # Create dim_customers
            dim_customers = customers_df.select(
                col("customer_id"),
                col("first_name"),
                col("last_name"),
                col("email"),
                col("city"),
                col("state"),
                col("country"),
                col("customer_segment"),
                col("registration_date"),
                lit(True).alias("is_active"),
                col("registration_date").alias("effective_date"),
                lit(True).alias("current_flag"),
                current_timestamp().alias("created_at")
            )
            
            dim_customers.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.processed_data.dim_customers") \
                .saveAsTable("processed_data.dim_customers")
            
            # Create dim_products
            dim_products = products_df.select(
                col("product_id"),
                col("product_name"),
                col("category"),
                col("subcategory"),
                col("brand"),
                col("price"),
                col("cost"),
                round(((col("price") - col("cost")) / col("price") * 100), 2).alias("margin_percentage"),
                col("is_active"),
                col("created_at").alias("effective_date"),
                lit(True).alias("current_flag"),
                current_timestamp().alias("created_at")
            )
            
            dim_products.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.processed_data.dim_products") \
                .saveAsTable("processed_data.dim_products")
            
            # Create dim_date
            self._create_date_dimension()
            
            self.logger.info("Dimension tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create dimension tables: {str(e)}")
            raise
    
    def _create_date_dimension(self):
        """Create date dimension table"""
        try:
            # Generate date range
            start_date = "2020-01-01"
            end_date = "2030-12-31"
            
            # Create date dimension using Spark SQL
            date_dim_sql = f"""
            SELECT 
                CAST(UNIX_TIMESTAMP(date_series) AS INT) as date_key,
                date_series as full_date,
                YEAR(date_series) as year,
                QUARTER(date_series) as quarter,
                MONTH(date_series) as month,
                MONTHNAME(date_series) as month_name,
                DAY(date_series) as day,
                DAYNAME(date_series) as day_name,
                WEEKOFYEAR(date_series) as week_of_year,
                CASE WHEN DAYOFWEEK(date_series) IN (1, 7) THEN true ELSE false END as is_weekend,
                false as is_holiday,
                null as holiday_name
            FROM (
                SELECT EXPLODE(SEQUENCE(TO_DATE('{start_date}'), TO_DATE('{end_date}'), INTERVAL 1 DAY)) as date_series
            )
            """
            
            date_dim_df = self.spark.sql(date_dim_sql)
            
            date_dim_df.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.processed_data.dim_date") \
                .saveAsTable("processed_data.dim_date")
            
            self.logger.info("Date dimension created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create date dimension: {str(e)}")
            raise
    
    def create_fact_tables(self):
        """Create and populate fact tables"""
        try:
            # Load required tables
            orders_df = self.spark.table("raw_data.orders")
            order_items_df = self.spark.table("raw_data.order_items")
            dim_customers_df = self.spark.table("processed_data.dim_customers")
            dim_products_df = self.spark.table("processed_data.dim_products")
            dim_date_df = self.spark.table("processed_data.dim_date")
            
            # Create fact_sales
            fact_sales = order_items_df \
                .join(orders_df, "order_id", "inner") \
                .join(dim_customers_df, order_items_df.order_id == orders_df.order_id, "inner") \
                .join(dim_products_df, order_items_df.product_id == dim_products_df.product_id, "inner") \
                .join(dim_date_df, date_format(orders_df.order_date, "yyyy-MM-dd") == dim_date_df.full_date, "inner") \
                .select(
                    col("order_id"),
                    col("customer_key"),
                    col("product_key"),
                    col("date_key"),
                    col("quantity"),
                    col("unit_price"),
                    col("total_price").alias("total_amount"),
                    (col("total_price") * col("cost") / col("price")).alias("cost_amount"),
                    (col("total_price") - (col("total_price") * col("cost") / col("price"))).alias("profit_amount"),
                    col("margin_percentage"),
                    (col("total_price") * col("discount_percentage") / 100).alias("discount_amount"),
                    col("tax_amount"),
                    col("shipping_cost"),
                    col("order_date"),
                    current_timestamp().alias("created_at")
                )
            
            fact_sales.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.processed_data.fact_sales") \
                .saveAsTable("processed_data.fact_sales")
            
            self.logger.info("Fact tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create fact tables: {str(e)}")
            raise
    
    def create_analytics_tables(self):
        """Create analytics summary tables"""
        try:
            # Load fact table
            fact_sales_df = self.spark.table("processed_data.fact_sales")
            dim_date_df = self.spark.table("processed_data.dim_date")
            dim_customers_df = self.spark.table("processed_data.dim_customers")
            
            # Daily sales summary
            daily_summary = fact_sales_df \
                .join(dim_date_df, "date_key", "inner") \
                .groupBy("full_date") \
                .agg(
                    countDistinct("order_id").alias("total_orders"),
                    sum("total_amount").alias("total_revenue"),
                    sum("profit_amount").alias("total_profit"),
                    avg("total_amount").alias("avg_order_value"),
                    countDistinct("customer_key").alias("total_customers")
                ) \
                .withColumnRenamed("full_date", "date")
            
            daily_summary.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.analytics.daily_sales_summary") \
                .saveAsTable("analytics.daily_sales_summary")
            
            # Product performance
            product_performance = fact_sales_df \
                .join(self.spark.table("processed_data.dim_products"), "product_key", "inner") \
                .groupBy("product_id", "product_name", "category") \
                .agg(
                    sum("total_amount").alias("total_sales"),
                    sum("quantity").alias("total_quantity"),
                    countDistinct("order_id").alias("total_orders"),
                    avg("margin_percentage").alias("profit_margin"),
                    max("order_date").alias("last_sale_date")
                )
            
            product_performance.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.analytics.product_performance") \
                .saveAsTable("analytics.product_performance")
            
            # Customer segments
            customer_segments = fact_sales_df \
                .join(dim_customers_df, "customer_key", "inner") \
                .groupBy("customer_id", "customer_segment") \
                .agg(
                    sum("total_amount").alias("total_spent"),
                    countDistinct("order_id").alias("total_orders"),
                    avg("total_amount").alias("avg_order_value"),
                    max("order_date").alias("last_order_date")
                ) \
                .withColumn("lifetime_value", col("total_spent")) \
                .withColumn("days_since_last_order", 
                           datediff(current_date(), col("last_order_date")))
            
            customer_segments.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.analytics.customer_segments") \
                .saveAsTable("analytics.customer_segments")
            
            self.logger.info("Analytics tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create analytics tables: {str(e)}")
            raise
    
    def create_ml_features(self):
        """Create ML feature tables"""
        try:
            fact_sales_df = self.spark.table("processed_data.fact_sales")
            dim_customers_df = self.spark.table("processed_data.dim_customers")
            dim_products_df = self.spark.table("processed_data.dim_products")
            
            # Customer features
            customer_features = fact_sales_df \
                .join(dim_customers_df, "customer_key", "inner") \
                .groupBy("customer_id") \
                .agg(
                    sum("total_amount").alias("total_spent"),
                    countDistinct("order_id").alias("total_orders"),
                    avg("total_amount").alias("avg_order_value"),
                    countDistinct("product_key").alias("unique_products"),
                    max("order_date").alias("last_order_date"),
                    min("order_date").alias("first_order_date")
                ) \
                .withColumn("days_since_last_order", 
                           datediff(current_date(), col("last_order_date"))) \
                .withColumn("customer_lifespan_days", 
                           datediff(col("last_order_date"), col("first_order_date")))
            
            # Convert to long format for ML features
            customer_ml_features = customer_features \
                .select(
                    col("customer_id"),
                    lit("total_spent").alias("feature_name"),
                    col("total_spent").alias("feature_value")
                ) \
                .union(
                    customer_features.select(
                        col("customer_id"),
                        lit("total_orders").alias("feature_name"),
                        col("total_orders").alias("feature_value")
                    )
                ) \
                .union(
                    customer_features.select(
                        col("customer_id"),
                        lit("avg_order_value").alias("feature_name"),
                        col("avg_order_value").alias("feature_value")
                    )
                ) \
                .union(
                    customer_features.select(
                        col("customer_id"),
                        lit("unique_products").alias("feature_name"),
                        col("unique_products").alias("feature_value")
                    )
                ) \
                .union(
                    customer_features.select(
                        col("customer_id"),
                        lit("days_since_last_order").alias("feature_name"),
                        col("days_since_last_order").alias("feature_value")
                    )
                )
            
            customer_ml_features.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.ml_features.customer_features") \
                .saveAsTable("ml_features.customer_features")
            
            # Product features
            product_features = fact_sales_df \
                .join(dim_products_df, "product_key", "inner") \
                .groupBy("product_id") \
                .agg(
                    sum("total_amount").alias("total_sales"),
                    sum("quantity").alias("total_quantity_sold"),
                    countDistinct("order_id").alias("total_orders"),
                    countDistinct("customer_key").alias("unique_customers"),
                    avg("margin_percentage").alias("avg_margin")
                )
            
            # Convert to long format
            product_ml_features = product_features \
                .select(
                    col("product_id"),
                    lit("total_sales").alias("feature_name"),
                    col("total_sales").alias("feature_value")
                ) \
                .union(
                    product_features.select(
                        col("product_id"),
                        lit("total_quantity_sold").alias("feature_name"),
                        col("total_quantity_sold").alias("feature_value")
                    )
                ) \
                .union(
                    product_features.select(
                        col("product_id"),
                        lit("total_orders").alias("feature_name"),
                        col("total_orders").alias("feature_value")
                    )
                ) \
                .union(
                    product_features.select(
                        col("product_id"),
                        lit("unique_customers").alias("feature_name"),
                        col("unique_customers").alias("feature_value")
                    )
                ) \
                .union(
                    product_features.select(
                        col("product_id"),
                        lit("avg_margin").alias("feature_name"),
                        col("avg_margin").alias("feature_value")
                    )
                )
            
            product_ml_features.write \
                .mode("overwrite") \
                .option("path", "/tmp/spark-warehouse/ecommerce_warehouse.ml_features.product_features") \
                .saveAsTable("ml_features.product_features")
            
            self.logger.info("ML features created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create ML features: {str(e)}")
            raise
    
    def run_advanced_analytics(self) -> Dict[str, Any]:
        """Run advanced analytics queries"""
        try:
            analytics_results = {}
            
            # Sales trends analysis
            sales_trends = self.spark.sql("""
                SELECT 
                    year,
                    month,
                    month_name,
                    SUM(total_revenue) as total_revenue,
                    SUM(total_orders) as total_orders,
                    AVG(avg_order_value) as avg_order_value
                FROM analytics.daily_sales_summary ds
                JOIN processed_data.dim_date dd ON ds.date = dd.full_date
                GROUP BY year, month, month_name
                ORDER BY year, month
            """)
            
            analytics_results['sales_trends'] = sales_trends.collect()
            
            # Customer cohort analysis
            customer_cohorts = self.spark.sql("""
                WITH customer_first_purchase AS (
                    SELECT 
                        customer_id,
                        MIN(order_date) as first_purchase_date
                    FROM processed_data.fact_sales fs
                    JOIN processed_data.dim_customers dc ON fs.customer_key = dc.customer_key
                    GROUP BY customer_id
                ),
                customer_orders AS (
                    SELECT 
                        fs.customer_key,
                        fs.order_date,
                        cfp.first_purchase_date,
                        DATEDIFF(fs.order_date, cfp.first_purchase_date) as days_since_first_purchase
                    FROM processed_data.fact_sales fs
                    JOIN processed_data.dim_customers dc ON fs.customer_key = dc.customer_key
                    JOIN customer_first_purchase cfp ON dc.customer_id = cfp.customer_id
                )
                SELECT 
                    YEAR(first_purchase_date) as cohort_year,
                    MONTH(first_purchase_date) as cohort_month,
                    COUNT(DISTINCT customer_key) as cohort_size,
                    COUNT(DISTINCT CASE WHEN days_since_first_purchase = 0 THEN customer_key END) as month_0,
                    COUNT(DISTINCT CASE WHEN days_since_first_purchase BETWEEN 1 AND 30 THEN customer_key END) as month_1,
                    COUNT(DISTINCT CASE WHEN days_since_first_purchase BETWEEN 31 AND 60 THEN customer_key END) as month_2
                FROM customer_orders
                GROUP BY YEAR(first_purchase_date), MONTH(first_purchase_date)
                ORDER BY cohort_year, cohort_month
            """)
            
            analytics_results['customer_cohorts'] = customer_cohorts.collect()
            
            # Product performance analysis
            product_analysis = self.spark.sql("""
                SELECT 
                    category,
                    COUNT(*) as product_count,
                    AVG(total_sales) as avg_sales,
                    SUM(total_sales) as total_category_sales,
                    AVG(profit_margin) as avg_margin
                FROM analytics.product_performance
                GROUP BY category
                ORDER BY total_category_sales DESC
            """)
            
            analytics_results['product_analysis'] = product_analysis.collect()
            
            # RFM Analysis
            rfm_analysis = self.spark.sql("""
                WITH customer_metrics AS (
                    SELECT 
                        customer_id,
                        MAX(order_date) as last_order_date,
                        COUNT(DISTINCT order_id) as frequency,
                        SUM(total_amount) as monetary_value
                    FROM processed_data.fact_sales fs
                    JOIN processed_data.dim_customers dc ON fs.customer_key = dc.customer_key
                    GROUP BY customer_id
                ),
                rfm_scores AS (
                    SELECT 
                        customer_id,
                        DATEDIFF(CURRENT_DATE(), last_order_date) as recency,
                        frequency,
                        monetary_value,
                        NTILE(5) OVER (ORDER BY DATEDIFF(CURRENT_DATE(), last_order_date) DESC) as r_score,
                        NTILE(5) OVER (ORDER BY frequency) as f_score,
                        NTILE(5) OVER (ORDER BY monetary_value) as m_score
                    FROM customer_metrics
                )
                SELECT 
                    CONCAT(r_score, f_score, m_score) as rfm_segment,
                    COUNT(*) as customer_count,
                    AVG(recency) as avg_recency,
                    AVG(frequency) as avg_frequency,
                    AVG(monetary_value) as avg_monetary_value
                FROM rfm_scores
                GROUP BY CONCAT(r_score, f_score, m_score)
                ORDER BY customer_count DESC
            """)
            
            analytics_results['rfm_analysis'] = rfm_analysis.collect()
            
            return analytics_results
            
        except Exception as e:
            self.logger.error(f"Failed to run advanced analytics: {str(e)}")
            return {}
    
    def optimize_tables(self):
        """Optimize tables for better performance"""
        try:
            # Analyze tables for statistics
            tables = [
                "processed_data.dim_customers",
                "processed_data.dim_products", 
                "processed_data.dim_date",
                "processed_data.fact_sales",
                "analytics.daily_sales_summary",
                "analytics.product_performance",
                "analytics.customer_segments"
            ]
            
            for table in tables:
                self.spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS")
                self.logger.info(f"Statistics computed for {table}")
            
            # Create indexes on key columns
            self.spark.sql("""
                CREATE INDEX IF NOT EXISTS idx_fact_sales_date_key 
                ON processed_data.fact_sales (date_key)
            """)
            
            self.spark.sql("""
                CREATE INDEX IF NOT EXISTS idx_fact_sales_customer_key 
                ON processed_data.fact_sales (customer_key)
            """)
            
            self.spark.sql("""
                CREATE INDEX IF NOT EXISTS idx_fact_sales_product_key 
                ON processed_data.fact_sales (product_key)
            """)
            
            self.logger.info("Table optimization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize tables: {str(e)}")
            raise
    
    def run_complete_pipeline(self):
        """Run complete data warehouse pipeline"""
        try:
            self.logger.info("Starting complete data warehouse pipeline")
            
            # Load raw data from PostgreSQL
            tables_to_load = ['customers', 'products', 'orders', 'order_items', 'reviews']
            for table in tables_to_load:
                self.load_data_from_postgres(table)
            
            # Create dimension tables
            self.create_dimension_tables()
            
            # Create fact tables
            self.create_fact_tables()
            
            # Create analytics tables
            self.create_analytics_tables()
            
            # Create ML features
            self.create_ml_features()
            
            # Optimize tables
            self.optimize_tables()
            
            # Run advanced analytics
            analytics_results = self.run_advanced_analytics()
            
            self.logger.info("Data warehouse pipeline completed successfully")
            return analytics_results
            
        except Exception as e:
            self.logger.error(f"Data warehouse pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()

# Example usage
if __name__ == "__main__":
    config = {
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_db': 'ecommerce_analytics',
        'postgres_user': 'admin',
        'postgres_password': 'password',
        'warehouse_dir': '/tmp/spark-warehouse'
    }
    
    warehouse = SparkDataWarehouse(config)
    results = warehouse.run_complete_pipeline()
    print("Pipeline completed successfully!")
